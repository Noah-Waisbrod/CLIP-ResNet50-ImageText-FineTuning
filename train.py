import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
import json

from model import CLIPModel
from cache_loader import create_clip_dataloaders


class CLIPTrainer:
    """
    Trainer class for fine-tuning CLIP image encoder.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_epochs=10,
        device="cuda",
        checkpoint_dir="./checkpoints",
        log_dir="./runs",
    ):
        """
        Args:
            model: CLIPModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            num_epochs: Number of training epochs
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device

        # Setup optimizer (only for trainable parameters)
        self.optimizer = optim.AdamW(
            model.get_trainable_params(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

        # Learning rate scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs * len(train_loader), eta_min=1e-6
        )

        # Setup logging
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"{log_dir}/clip_{timestamp}")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        num_batches = 0

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs}"
        )

        for batch_idx, (images, text_embeddings, _) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            text_embeddings = text_embeddings.to(self.device)

            # Forward pass
            image_embs, text_embs, logits = self.model(images, text_embeddings)

            # Compute loss
            loss = self.model.compute_loss(logits)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_params(), max_norm=1.0
            )

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss/num_batches:.4f}",
                    "lr": f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                }
            )

            self.global_step += 1

        # Log to tensorboard
        self.writer.add_scalar("train/loss", loss.item(), self.global_step)
        self.writer.add_scalar(
            "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
        )

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()

        total_loss = 0
        num_batches = 0

        # Metrics for retrieval evaluation
        correct_i2t = 0  # Image-to-text retrieval accuracy
        correct_t2i = 0  # Text-to-image retrieval accuracy
        total_samples = 0

        pbar = tqdm(self.val_loader, desc="Validation")

        # Accumulate logit statistics
        all_diag_values = []
        all_offdiag_values = []

        for images, text_embeddings, _ in pbar:
            # Move to device
            images = images.to(self.device)
            text_embeddings = text_embeddings.to(self.device)

            # Forward pass
            image_embs, text_embs, logits = self.model(images, text_embeddings)

            self.writer.add_scalar(
                "train/logits_mean", logits.mean().item(), self.global_step
            )
            self.writer.add_scalar(
                "train/logits_std", logits.std().item(), self.global_step
            )

            # Compute loss
            loss = self.model.compute_loss(logits)

            total_loss += loss.item()
            num_batches += 1

            # Compute retrieval accuracy
            batch_size = images.shape[0]

            # Image-to-text: for each image, is the correct text ranked first?
            i2t_preds = logits.argmax(dim=1)  # (B,)
            correct_i2t += (
                (i2t_preds == torch.arange(batch_size, device=self.device)).sum().item()
            )

            # Text-to-image: for each text, is the correct image ranked first?
            t2i_preds = logits.argmax(dim=0)  # (B,)
            correct_t2i += (
                (t2i_preds == torch.arange(batch_size, device=self.device)).sum().item()
            )

            # Collect logit statistics
            batch_size = logits.shape[0]
            diag_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
            all_diag_values.append(logits[diag_mask])
            all_offdiag_values.append(logits[~diag_mask])

            total_samples += batch_size

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "i2t_acc": f"{correct_i2t/total_samples:.4f}",
                    "t2i_acc": f"{correct_t2i/total_samples:.4f}",
                }
            )

        avg_loss = total_loss / num_batches
        i2t_accuracy = correct_i2t / total_samples
        t2i_accuracy = correct_t2i / total_samples

        # Log validation metrics
        # Log detailed statistics
        diag_tensor = torch.cat(all_diag_values)
        offdiag_tensor = torch.cat(all_offdiag_values)

        print(f"\n  Logit Statistics:")
        print(
            f"    Diagonal (correct pairs): mean={diag_tensor.mean():.4f}, std={diag_tensor.std():.4f}"
        )
        print(
            f"    Off-diagonal (incorrect): mean={offdiag_tensor.mean():.4f}, std={offdiag_tensor.std():.4f}"
        )
        print(f"    Separation: {(diag_tensor.mean() - offdiag_tensor.mean()):.4f}")

        self.writer.add_scalar(
            "val/logits_diagonal_mean", diag_tensor.mean(), self.global_step
        )
        self.writer.add_scalar(
            "val/logits_offdiag_mean", offdiag_tensor.mean(), self.global_step
        )
        self.writer.add_scalar(
            "val/logits_separation",
            diag_tensor.mean() - offdiag_tensor.mean(),
            self.global_step,
        )

        self.writer.add_scalar("val/loss", avg_loss, self.global_step)
        self.writer.add_scalar("val/i2t_accuracy", i2t_accuracy, self.global_step)
        self.writer.add_scalar("val/t2i_accuracy", t2i_accuracy, self.global_step)

        return avg_loss, i2t_accuracy, t2i_accuracy

    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Checkpoint loaded from: {filepath}")
        print(f"  Resuming from epoch {self.current_epoch}")
        print(f"  Global step: {self.global_step}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting CLIP Training")
        print("=" * 60 + "\n")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, i2t_acc, t2i_acc = self.validate()

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Image→Text Accuracy: {i2t_acc:.4f}")
            print(f"  Text→Image Accuracy: {t2i_acc:.4f}")

            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", is_best=True)
                print(f"  ✓ New best validation loss!")

            print()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60 + "\n")

        self.writer.close()


def main():
    """Main training script."""

    # Configuration
    CONFIG = {
        "data_root": "./data/coco2014",
        "cache_root": "./data/cache",
        "batch_size": 64,
        "num_workers": 4,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "num_epochs": 10,
        "subset_ratio": 0.1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_dir": "./checkpoints",
        "log_dir": "./runs",
    }

    print("=" * 60)
    print("CLIP Fine-tuning Configuration")
    print("=" * 60)
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    # Save configuration
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    with open(os.path.join(CONFIG["checkpoint_dir"], "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = create_clip_dataloaders(
        data_root=CONFIG["data_root"],
        cache_root=CONFIG["cache_root"],
        train_split="train2014",
        val_split="val2014",
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        subset_ratio=CONFIG["subset_ratio"],
    )

    # Create model
    print("\nInitializing model...")
    model = CLIPModel(embedding_dim=512, temperature=0.07)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_params())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = CLIPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        num_epochs=CONFIG["num_epochs"],
        device=CONFIG["device"],
        checkpoint_dir=CONFIG["checkpoint_dir"],
        log_dir=CONFIG["log_dir"],
    )

    # Train
    trainer.train()

    print("\n✓ Training completed successfully!")
    print(f"Checkpoints saved to: {CONFIG['checkpoint_dir']}")
    print(f"TensorBoard logs saved to: {CONFIG['log_dir']}")
    print(f"\nTo view training progress, run:")
    print(f"  tensorboard --logdir={CONFIG['log_dir']}")


if __name__ == "__main__":
    main()
