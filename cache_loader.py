import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms


class CLIPFinetuneDataset(Dataset):
    """
    Dataset for fine-tuning CLIP image encoder with precomputed text embeddings.
    Each image may have multiple captions (COCO has ~5 per image).
    """

    def __init__(
        self, data_root="./data/coco2014", cache_root="./data/cache", split="train2014"
    ):
        """
        Args:
            data_root: Root directory containing COCO images
            cache_root: Directory containing cached text embeddings
            split: Dataset split (e.g., 'train2014', 'val2014')
        """
        self.data_root = data_root
        self.cache_root = cache_root
        self.split = split

        # Load precomputed text embeddings
        emb_file = os.path.join(cache_root, f"{split}_caption_embs.pt")
        index_file = os.path.join(cache_root, f"{split}_index.json")

        if not os.path.exists(emb_file) or not os.path.exists(index_file):
            raise FileNotFoundError(
                f"Cached embeddings not found. Please run preprocessing first.\n"
                f"Looking for: {emb_file} and {index_file}"
            )

        print(f"Loading cached embeddings from {emb_file}...")
        self.text_embeddings = torch.load(emb_file)  # Shape: (N, 512)

        print(f"Loading index from {index_file}...")
        with open(index_file, "r") as f:
            self.index = json.load(f)

        print(f"Loaded {len(self.index)} image-caption pairs for {split}")

        # CLIP normalization constants
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Returns:
            image: Normalized image tensor (3, 224, 224)
            text_embedding: Precomputed CLIP text embedding (512,)
            image_path: Path to the image file (for debugging/visualization)
        """
        sample = self.index[idx]
        img_path = sample["image_path"]
        emb_idx = sample["embedding_index"]

        # Load and transform image
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            image = torch.zeros(3, 224, 224)

        # Get precomputed text embedding
        text_embedding = self.text_embeddings[emb_idx]

        return image, text_embedding, img_path


def create_clip_dataloaders(
    data_root="./data/coco2014",
    cache_root="./data/cache",
    train_split="train2014",
    val_split="val2014",
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    subset_ratio=1.0,
):
    """
    Create train and validation dataloaders for CLIP fine-tuning.

    Args:
        data_root: Root directory containing COCO images
        cache_root: Directory containing cached text embeddings
        train_split: Training split name
        val_split: Validation split name
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader
    """

    # Create datasets
    train_dataset = CLIPFinetuneDataset(
        data_root=data_root, cache_root=cache_root, split=train_split
    )

    val_dataset = CLIPFinetuneDataset(
        data_root=data_root, cache_root=cache_root, split=val_split
    )

    if subset_ratio < 1.0:
        import math

        train_len = math.ceil(len(train_dataset) * subset_ratio)
        val_len = math.ceil(len(val_dataset) * subset_ratio)

        train_dataset = Subset(train_dataset, list(range(train_len)))
        val_dataset = Subset(val_dataset, list(range(val_len)))

        print(f"⚠️ Using subset: {train_len} train samples, {val_len} val samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for consistent batch size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def verify_text_embeddings(dataset, num_samples=5):
    """
    Verify text embeddings by re-encoding captions and comparing with cached embeddings.

    Args:
        dataset: CLIPFinetuneDataset instance
        num_samples: Number of random samples to verify

    Returns:
        bool: True if embeddings match within tolerance
    """
    import random
    from transformers import CLIPTokenizer, CLIPTextModel

    print(f"\n{'='*60}")
    print(f"Text Embedding Verification")
    print(f"{'='*60}\n")

    # Load CLIP text encoder for verification
    print("Loading CLIP text encoder for verification...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder.eval()

    # Load original annotations to get captions
    ann_file = os.path.join(
        dataset.data_root, "annotations", f"captions_{dataset.split}.json"
    )
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    # Create mapping from image path to captions
    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    filename_to_captions = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        filename = id_to_filename[img_id]
        if filename not in filename_to_captions:
            filename_to_captions[filename] = []
        filename_to_captions[filename].append(ann["caption"])

    # Sample random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    all_match = True
    max_diff = 0.0

    for idx in indices:
        image, cached_emb, img_path = dataset[idx]
        filename = os.path.basename(img_path)

        # Get captions for this image
        captions = filename_to_captions.get(filename, [])
        if not captions:
            print(f"⚠️  No captions found for {filename}")
            continue

        print(f"\nSample {idx}: {filename}")
        print(f"  Number of captions: {len(captions)}")

        # Re-encode all captions for this image
        with torch.no_grad():
            tokens = tokenizer(
                captions, padding=True, truncation=True, return_tensors="pt"
            )
            output = text_encoder(**tokens)
            fresh_embs = output.last_hidden_state[:, 0, :]  # (num_captions, 512)

        # Find which caption matches the cached embedding
        best_match_idx = -1
        min_distance = float("inf")

        for i, fresh_emb in enumerate(fresh_embs):
            distance = torch.norm(cached_emb - fresh_emb).item()
            if distance < min_distance:
                min_distance = distance
                best_match_idx = i

        max_diff = max(max_diff, min_distance)

        # Check if match is close enough (tolerance for floating point differences)
        if min_distance < 1e-4:
            print(
                f"  ✓ Embedding matches caption {best_match_idx}: '{captions[best_match_idx][:60]}...'"
            )
            print(f"    L2 distance: {min_distance:.2e}")
        else:
            print(f"  ✗ Embedding mismatch! Closest caption {best_match_idx}")
            print(f"    L2 distance: {min_distance:.2e}")
            print(f"    Caption: '{captions[best_match_idx][:60]}...'")
            all_match = False

        # Show embedding statistics
        print(
            f"  Cached embedding stats: mean={cached_emb.mean():.4f}, std={cached_emb.std():.4f}, norm={torch.norm(cached_emb):.4f}"
        )

    print(f"\n{'='*60}")
    print(f"Verification Summary:")
    print(f"  Maximum L2 distance: {max_diff:.2e}")
    print(f"  All embeddings valid: {'✓ YES' if all_match else '✗ NO'}")
    print(f"{'='*60}\n")

    return all_match


def verify_dataset_integrity(dataset, num_samples=5):
    """
    Verify dataset by displaying random image-caption pairs with their captions.

    Args:
        dataset: CLIPFinetuneDataset instance
        num_samples: Number of random samples to display
    """
    import random
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"\n{'='*60}")
    print(f"Dataset Integrity Check - Displaying {num_samples} random samples")
    print(f"{'='*60}\n")

    # Load annotations to get captions
    ann_file = os.path.join(
        dataset.data_root, "annotations", f"captions_{dataset.split}.json"
    )
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    # Create mapping from image path to captions
    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    filename_to_captions = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        filename = id_to_filename[img_id]
        if filename not in filename_to_captions:
            filename_to_captions[filename] = []
        filename_to_captions[filename].append(ann["caption"])

    indices = random.sample(range(len(dataset)), num_samples)

    # Inverse normalization for visualization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 5))
    if num_samples == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        image, text_emb, img_path = dataset[idx]
        filename = os.path.basename(img_path)

        # Denormalize image for display
        image_display = image * std + mean
        image_display = torch.clamp(image_display, 0, 1)
        image_np = image_display.permute(1, 2, 0).numpy()

        # Get captions
        captions = filename_to_captions.get(filename, ["No caption found"])
        caption_text = (
            captions[0][:50] + "..." if len(captions[0]) > 50 else captions[0]
        )

        # Display
        axes[i].imshow(image_np)
        axes[i].axis("off")
        axes[i].set_title(
            f"{caption_text}\n({len(captions)} captions)", fontsize=8, wrap=True
        )

        # Print info
        print(f"Sample {idx}:")
        print(f"  Image: {filename}")
        print(f"  Image shape: {image.shape}")
        print(f"  Text embedding shape: {text_emb.shape}")
        print(f"  Text embedding norm: {torch.norm(text_emb):.4f}")
        print(f"  Number of captions: {len(captions)}")
        print(f"  First caption: '{captions[0]}'")
        print()

    plt.tight_layout()
    # plt.savefig("dataset_verification.png", dpi=150, bbox_inches="tight")
    print(f"Visualization saved to 'dataset_verification.png'")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Configuration
    DATA_ROOT = "./data/coco2014"
    CACHE_ROOT = "./data/cache"
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    print("Creating dataloaders...")
    train_loader, val_loader = create_clip_dataloaders(
        data_root=DATA_ROOT,
        cache_root=CACHE_ROOT,
        train_split="train2014",
        val_split="val2014",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"\nDataloader Statistics:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Batch size: {BATCH_SIZE}")

    # Test loading a batch
    print(f"\nTesting batch loading...")
    images, text_embeddings, img_paths = next(iter(train_loader))
    print(f"  Batch image shape: {images.shape}")
    print(f"  Batch text embedding shape: {text_embeddings.shape}")
    print(f"  Number of paths: {len(img_paths)}")
    print(f"  Image tensor range: [{images.min():.3f}, {images.max():.3f}]")

    # Verify dataset integrity (images + captions)
    print(f"\nVerifying dataset integrity...")
    verify_dataset_integrity(train_loader.dataset, num_samples=5)

    # Verify text embeddings match captions
    print(f"\nVerifying text embeddings...")
    embeddings_valid = verify_text_embeddings(train_loader.dataset, num_samples=5)

    if embeddings_valid:
        print("\n✓ All verifications passed! Dataset is ready for training.")
    else:
        print("\n⚠️  Some embeddings don't match. Consider re-running preprocessing.")
