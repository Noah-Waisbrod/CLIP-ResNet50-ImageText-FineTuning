import os
import json
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel


class CLIPCaptionPreprocessor:
    def __init__(self, data_root="./data/coco2014", cache_root="./data/cache"):
        self.data_root = data_root
        self.cache_root = cache_root
        os.makedirs(cache_root, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Use full CLIP model (more reliable than separate components)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.model.to(self.device)
        self.model.eval()

    def load_annotations(self, split):
        ann_file = os.path.join(self.data_root, "annotations", f"captions_{split}.json")

        with open(ann_file, "r") as f:
            coco = json.load(f)

        id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

        # Group captions by image
        image_captions = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_captions:
                image_captions[img_id] = []
            image_captions[img_id].append(ann["caption"])

        # Create samples: one caption per image (use first caption)
        samples = []
        for img_id, captions in image_captions.items():
            filename = id_to_filename[img_id]
            img_path = os.path.join(self.data_root, "images", split, filename)
            caption = captions[0]  # Use first caption consistently
            samples.append((img_path, caption))

        print(f"[INFO] Loaded {len(samples)} image-caption pairs for {split}")
        return samples

    @torch.no_grad()
    def encode_caption_batch(self, captions):
        """Encode a batch of captions using CLIP text encoder."""
        # Tokenize
        tokens = self.tokenizer(
            captions, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )

        # Move to device
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # Encode using CLIP's get_text_features
        embeddings = self.model.get_text_features(**tokens)

        # Explicitly normalize (make sure they're unit vectors)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu()

    def preprocess_split(self, split, batch_size=256):
        pairs = self.load_annotations(split)

        emb_file = os.path.join(self.cache_root, f"{split}_caption_embs.pt")
        index_file = os.path.join(self.cache_root, f"{split}_index.json")

        # Check if already processed
        if os.path.exists(emb_file) and os.path.exists(index_file):
            print(f"[CACHE] {split} already processed.")
            response = input("Delete and reprocess? (y/n): ")
            if response.lower() != "y":
                return
            else:
                os.remove(emb_file)
                os.remove(index_file)

        print(f"[PROCESS] Encoding {len(pairs)} captions for {split}...")

        total = len(pairs)
        all_embeddings = []
        index_list = []

        # Process in batches
        for i in tqdm(range(0, total, batch_size), desc=f"Processing {split}"):
            chunk = pairs[i : i + batch_size]

            # Extract captions and paths from chunk
            img_paths = [p for p, _ in chunk]
            captions = [c for _, c in chunk]

            # Encode batch of captions
            batch_embeddings = self.encode_caption_batch(captions)  # (batch_size, 512)
            all_embeddings.append(batch_embeddings)

            # Create index entries for this batch
            for j, (img_path, caption) in enumerate(zip(img_paths, captions)):
                global_idx = i + j  # Global index in the full dataset
                index_list.append(
                    {
                        "image_path": img_path,
                        "embedding_index": global_idx,
                        "caption": caption,
                    }
                )

        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Verify embeddings
        print(f"\n[VERIFY] Embedding statistics:")
        print(f"  Shape: {all_embeddings.shape}")
        print(f"  Mean: {all_embeddings.mean():.6f}")
        print(f"  Std: {all_embeddings.std():.6f}")
        print(f"  Norm (first 3): {torch.norm(all_embeddings[:3], dim=1)}")

        # Check diversity
        sample_size = min(100, len(all_embeddings))
        pairwise = torch.matmul(
            all_embeddings[:sample_size], all_embeddings[:sample_size].T
        )
        off_diag_mask = ~torch.eye(sample_size, dtype=torch.bool)
        off_diag = pairwise[off_diag_mask]

        print(f"  Pairwise similarity (first {sample_size}):")
        print(f"    Mean: {off_diag.mean():.6f}")
        print(f"    Min: {off_diag.min():.6f}")
        print(f"    Max: {off_diag.max():.6f}")

        # Show first 3 embeddings
        print(f"  First 3 embeddings (first 10 dims):")
        print(all_embeddings[:3, :10])

        # Show first 3 captions for reference
        print(f"\n  First 3 captions:")
        for idx in range(min(3, len(index_list))):
            print(f"    [{idx}] {index_list[idx]['caption'][:80]}")

        if off_diag.mean() > 0.95:
            print("\n  ⚠️  WARNING: Embeddings are very similar! Something is wrong.")
            return

        if off_diag.mean() < 0.3 or off_diag.mean() > 0.8:
            print(
                f"\n  ⚠️  WARNING: Unusual similarity mean ({off_diag.mean():.3f}). Expected 0.4-0.7"
            )

        # Save embeddings
        print(f"\n[SAVE] Saving to {emb_file}...")
        torch.save(all_embeddings, emb_file)

        # Save index
        print(f"[SAVE] Saving to {index_file}...")
        with open(index_file, "w") as f:
            json.dump(index_list, f)

        print(
            f"\n[DONE] Successfully saved {len(all_embeddings)} embeddings for {split}"
        )


if __name__ == "__main__":
    DATA_ROOT = "./data/coco2014"
    CACHE_ROOT = "./data/cache"
    BATCH_SIZE = 256

    preprocessor = CLIPCaptionPreprocessor(data_root=DATA_ROOT, cache_root=CACHE_ROOT)

    # Process both splits
    for split in ["train2014", "val2014"]:
        print(f"\n{'='*60}")
        print(f"Processing {split}")
        print(f"{'='*60}")
        preprocessor.preprocess_split(split=split, batch_size=BATCH_SIZE)
        print()
