import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from collections import defaultdict

from model import CLIPModel
from cache_loader import CLIPFinetuneDataset
from transformers import CLIPTokenizer, CLIPModel as HFCLIPModel


class CLIPEvaluator:
    """
    Comprehensive evaluation suite for fine-tuned CLIP model.
    """

    def __init__(self, model, val_dataset, device="cuda", checkpoint_path=None):
        """
        Args:
            model: CLIPModel instance
            val_dataset: Validation dataset
            device: Device to run evaluation on
            checkpoint_path: Path to model checkpoint (optional)
        """
        self.model = model.to(device)
        self.val_dataset = val_dataset
        self.device = device

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded from epoch {checkpoint['epoch']}")

        self.model.eval()

        # Load text encoder for custom queries
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = HFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model.to(device)
        self.text_model.eval()

        print(f"Evaluator initialized with {len(val_dataset)} validation samples")

    @torch.no_grad()
    def encode_text(self, texts):
        """Encode text queries using pretrained CLIP text encoder."""
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self.device)

        text_features = self.text_model.get_text_features(**tokens)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
        return text_features

    @torch.no_grad()
    def compute_embeddings(self, subset_size=None):
        """
        Compute image and text embeddings for the entire validation set.

        Args:
            subset_size: If provided, only use first N samples (for faster testing)

        Returns:
            image_embeddings: (N, 512) tensor
            text_embeddings: (N, 512) tensor
            image_paths: List of image paths
        """
        print("\nComputing embeddings for validation set...")

        # Determine subset
        if subset_size is not None:
            indices = list(range(min(subset_size, len(self.val_dataset))))
        else:
            indices = list(range(len(self.val_dataset)))

        image_embeddings = []
        text_embeddings = []
        image_paths = []

        batch_size = 128
        for i in tqdm(range(0, len(indices), batch_size), desc="Encoding"):
            batch_indices = indices[i : i + batch_size]

            images_batch = []
            texts_batch = []
            paths_batch = []

            for idx in batch_indices:
                image, text_emb, img_path = self.val_dataset[idx]
                images_batch.append(image)
                texts_batch.append(text_emb)
                paths_batch.append(img_path)

            # Stack batch
            images_batch = torch.stack(images_batch).to(self.device)
            texts_batch = torch.stack(texts_batch).to(self.device)

            # Encode images
            img_embs = self.model.image_encoder(images_batch)

            image_embeddings.append(img_embs.cpu())
            text_embeddings.append(texts_batch.cpu())
            image_paths.extend(paths_batch)

        image_embeddings = torch.cat(image_embeddings, dim=0)
        text_embeddings = torch.cat(text_embeddings, dim=0)

        print(f"  Computed {len(image_embeddings)} image-text pairs")
        return image_embeddings, text_embeddings, image_paths

    def compute_recall_at_k(self, similarities, k_values=[1, 5, 10]):
        """
        Compute Recall@K given a similarity matrix.

        Args:
            similarities: (N, N) similarity matrix
            k_values: List of K values to compute recall for

        Returns:
            dict: {k: recall_value}
        """
        N = similarities.shape[0]
        ranks = torch.argsort(similarities, dim=1, descending=True)

        # Ground truth: diagonal (i-th query matches i-th item)
        ground_truth = torch.arange(N)

        recalls = {}
        for k in k_values:
            # Check if ground truth is in top-k
            top_k = ranks[:, :k]
            correct = (top_k == ground_truth.unsqueeze(1)).any(dim=1).float()
            recalls[k] = correct.mean().item()

        return recalls

    def evaluate_retrieval(self, image_embeddings, text_embeddings):
        """
        Evaluate image-to-text and text-to-image retrieval.

        Returns:
            dict: Results containing recall@k for both directions
        """
        print("\n" + "=" * 60)
        print("RETRIEVAL EVALUATION")
        print("=" * 60)

        # Compute similarity matrix
        print("\nComputing similarity matrix...")
        similarities = torch.matmul(image_embeddings, text_embeddings.T)  # (N, N)

        print(f"  Similarity matrix shape: {similarities.shape}")
        print(f"  Diagonal mean (correct pairs): {similarities.diagonal().mean():.4f}")
        print(
            f"  Off-diagonal mean (incorrect): {similarities[~torch.eye(len(similarities), dtype=bool)].mean():.4f}"
        )

        # Image-to-Text retrieval
        print("\n[Image → Text Retrieval]")
        i2t_recalls = self.compute_recall_at_k(similarities, k_values=[1, 5, 10])
        for k, recall in i2t_recalls.items():
            print(f"  Recall@{k}: {recall*100:.2f}%")

        # Text-to-Image retrieval
        print("\n[Text → Image Retrieval]")
        t2i_recalls = self.compute_recall_at_k(similarities.T, k_values=[1, 5, 10])
        for k, recall in t2i_recalls.items():
            print(f"  Recall@{k}: {recall*100:.2f}%")

        # Mean recall
        mean_recall = np.mean(
            [
                i2t_recalls[1],
                i2t_recalls[5],
                i2t_recalls[10],
                t2i_recalls[1],
                t2i_recalls[5],
                t2i_recalls[10],
            ]
        )
        print(f"\n[Mean Recall@(1,5,10)]: {mean_recall*100:.2f}%")

        results = {
            "image_to_text": i2t_recalls,
            "text_to_image": t2i_recalls,
            "mean_recall": mean_recall,
            "similarity_matrix": similarities,
        }

        return results

    @torch.no_grad()
    def text_to_image_search(self, query_text, image_embeddings, image_paths, top_k=5):
        """
        Given a text query, retrieve top-k most similar images.

        Args:
            query_text: String query (e.g., "sport")
            image_embeddings: Precomputed image embeddings
            image_paths: List of image paths
            top_k: Number of images to retrieve

        Returns:
            List of (image_path, similarity_score) tuples
        """
        # Encode query text
        query_embedding = self.encode_text([query_text])  # (1, 512)

        # Move image embeddings to same device
        image_embeddings = image_embeddings.to(self.device)

        # Compute similarities
        similarities = torch.matmul(
            query_embedding, image_embeddings.T
        ).squeeze()  # (N,)

        # Get top-k indices
        top_k_indices = torch.argsort(similarities, descending=True)[:top_k]

        results = []
        for idx in top_k_indices:
            results.append((image_paths[idx.item()], similarities[idx].item()))

        return results

    def visualize_text_to_image_retrieval(
        self, query_text, image_embeddings, image_paths, top_k=5
    ):
        """
        Visualize top-k retrieved images for a text query.
        """
        print(f"\n[Text Query]: '{query_text}'")
        print(f"Retrieving top-{top_k} images...")

        results = self.text_to_image_search(
            query_text, image_embeddings, image_paths, top_k
        )

        fig, axes = plt.subplots(1, top_k, figsize=(4 * top_k, 5))
        if top_k == 1:
            axes = [axes]

        for i, (img_path, score) in enumerate(results):
            try:
                # Normalize path separators for cross-platform compatibility
                img_path = img_path.replace("\\", "/")
                img = Image.open(img_path).convert("RGB")
                axes[i].imshow(img)
                axes[i].axis("off")
                axes[i].set_title(f"Rank {i+1}\nScore: {score:.3f}", fontsize=10)
            except Exception as e:
                print(f"  Warning: Could not load {os.path.basename(img_path)}")
                axes[i].text(
                    0.5,
                    0.5,
                    f"Error loading\n{os.path.basename(img_path)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                axes[i].axis("off")

        plt.suptitle(f"Text Query: '{query_text}'", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save figure
        save_path = f"retrieval_{query_text.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved visualization to: {save_path}")
        plt.show()

        return results

    @torch.no_grad()
    def zero_shot_classification(self, image_path, class_labels):
        """
        Classify an image given a list of class labels (zero-shot).

        Args:
            image_path: Path to image file
            class_labels: List of text labels (e.g., ['a person', 'an animal', 'a landscape'])

        Returns:
            dict: {label: probability}
        """
        # Normalize path separators
        image_path = image_path.replace("\\", "/")

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.val_dataset.transform(image).unsqueeze(0).to(self.device)

        # Encode image
        image_embedding = self.model.image_encoder(image_tensor)  # (1, 512)

        # Encode class labels as text
        text_embeddings = self.encode_text(class_labels)  # (num_classes, 512)

        # Compute similarities
        similarities = torch.matmul(
            image_embedding, text_embeddings.T
        ).squeeze()  # (num_classes,)

        # Convert to probabilities using softmax
        probabilities = torch.nn.functional.softmax(similarities / 0.07, dim=0)

        # Create results dict
        results = {
            label: prob.item() for label, prob in zip(class_labels, probabilities)
        }

        return results, similarities

    def visualize_classification(self, image_path, class_labels):
        """
        Visualize zero-shot classification results.
        """
        # Normalize path
        image_path = image_path.replace("\\", "/")

        print(f"\n[Zero-Shot Classification]")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Classes: {class_labels}")

        try:
            probs, similarities = self.zero_shot_classification(
                image_path, class_labels
            )
        except Exception as e:
            print(f"  Error: Could not process image - {e}")
            return None

        # Sort by probability
        sorted_results = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        print(f"\nResults:")
        for label, prob in sorted_results:
            print(f"  {label:20s}: {prob*100:5.2f}%")

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Show image
        img = Image.open(image_path).convert("RGB")
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title(f"Image: {os.path.basename(image_path)}", fontsize=12)

        # Show probabilities
        labels = [item[0] for item in sorted_results]
        probs_list = [item[1] * 100 for item in sorted_results]

        colors = ["green" if i == 0 else "skyblue" for i in range(len(labels))]
        ax2.barh(labels, probs_list, color=colors)
        ax2.set_xlabel("Probability (%)", fontsize=11)
        ax2.set_title("Classification Probabilities", fontsize=12)
        ax2.set_xlim(0, 100)

        # Add percentage labels
        for i, (label, prob) in enumerate(zip(labels, probs_list)):
            ax2.text(prob + 2, i, f"{prob:.1f}%", va="center", fontsize=10)

        plt.tight_layout()

        # Save
        save_path = (
            f"classification_{os.path.basename(image_path).replace('.jpg', '.png')}"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved visualization to: {save_path}")
        plt.show()

        return probs

    def run_full_evaluation(self, subset_size=None, save_results=True):
        """
        Run complete evaluation suite.

        Args:
            subset_size: If provided, only evaluate on subset (for speed)
            save_results: Whether to save results to JSON

        Returns:
            dict: Complete evaluation results
        """
        print("\n" + "=" * 60)
        print("FULL CLIP MODEL EVALUATION")
        print("=" * 60)

        # Compute embeddings
        image_embeddings, text_embeddings, image_paths = self.compute_embeddings(
            subset_size
        )

        # Evaluate retrieval
        retrieval_results = self.evaluate_retrieval(image_embeddings, text_embeddings)

        # Example text-to-image retrievals
        print("\n" + "=" * 60)
        print("TEXT-TO-IMAGE RETRIEVAL EXAMPLES")
        print("=" * 60)

        example_queries = ["sport", "food", "person", "animal", "car"]
        for query in example_queries:
            self.visualize_text_to_image_retrieval(
                query, image_embeddings, image_paths, top_k=5
            )

        # Example zero-shot classification
        print("\n" + "=" * 60)
        print("ZERO-SHOT CLASSIFICATION EXAMPLES")
        print("=" * 60)

        # Pick a few random images that actually exist
        import random

        class_sets = [
            ["a person", "an animal", "a landscape"],
            ["a dog", "a cat", "a bird", "a horse"],
            ["food", "furniture", "vehicle", "building"],
        ]

        # Find valid images
        valid_image_paths = []
        for path in image_paths:
            normalized_path = path.replace("\\", "/")
            if os.path.exists(normalized_path):
                valid_image_paths.append(normalized_path)

        if len(valid_image_paths) > 0:
            sample_indices = random.sample(
                range(len(valid_image_paths)), min(3, len(valid_image_paths))
            )

            for idx, classes in zip(sample_indices, class_sets):
                self.visualize_classification(valid_image_paths[idx], classes)
        else:
            print("Warning: No valid images found for classification examples")

        # Compile all results
        results = {
            "retrieval": {
                "image_to_text": retrieval_results["image_to_text"],
                "text_to_image": retrieval_results["text_to_image"],
                "mean_recall": retrieval_results["mean_recall"],
            },
            "num_samples": len(image_paths),
        }

        # Save results
        if save_results:
            results_file = "evaluation_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to: {results_file}")

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        return results


def main():
    """Main evaluation script."""

    # Configuration
    CONFIG = {
        "data_root": "./data/coco2014",
        "cache_root": "./data/cache",
        "checkpoint_path": "./checkpoints/best_model.pt",  # Path to your trained model
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "subset_size": 5000,  # Use subset for faster evaluation, None for full dataset
    }

    print("=" * 60)
    print("CLIP Model Evaluation Configuration")
    print("=" * 60)
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = CLIPFinetuneDataset(
        data_root=CONFIG["data_root"], cache_root=CONFIG["cache_root"], split="val2014"
    )

    # Initialize model
    print("\nInitializing model...")
    model = CLIPModel(embedding_dim=512, temperature=0.07)

    # Create evaluator
    evaluator = CLIPEvaluator(
        model=model,
        val_dataset=val_dataset,
        device=CONFIG["device"],
        checkpoint_path=CONFIG["checkpoint_path"],
    )

    # Run full evaluation
    results = evaluator.run_full_evaluation(
        subset_size=CONFIG["subset_size"], save_results=True
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nImage → Text Retrieval:")
    for k, v in results["retrieval"]["image_to_text"].items():
        print(f"  Recall@{k}: {v*100:.2f}%")

    print(f"\nText → Image Retrieval:")
    for k, v in results["retrieval"]["text_to_image"].items():
        print(f"  Recall@{k}: {v*100:.2f}%")

    print(f"\nMean Recall: {results['retrieval']['mean_recall']*100:.2f}%")
    print(f"Evaluated on {results['num_samples']} samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
