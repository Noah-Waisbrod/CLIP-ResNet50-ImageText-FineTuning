import torch
import torch.nn as nn
import torchvision.models as models


class CLIPImageEncoder(nn.Module):
    """
    CLIP Image Encoder using ResNet50 backbone with projection head.

    Architecture:
    - ResNet50 pretrained on ImageNet (frozen or trainable)
    - Projection head: Linear(2048) -> GELU -> Linear(512)
    - Output: 512-dimensional CLIP embedding space
    """

    def __init__(
        self,
        embedding_dim=512,
        pretrained=True,
        freeze_backbone=False,
    ):
        """
        Args:
            embedding_dim: Dimension of output CLIP embeddings (default: 512)
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze ResNet50 backbone parameters
        """
        super(CLIPImageEncoder, self).__init__()

        # Load pretrained ResNet50 (using new weights API)
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)

        # Remove the final classification layer
        # ResNet50 outputs 2048-dimensional features before the FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head: maps 2048-dim ResNet features to 512-dim CLIP space
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, images):
        features = self.backbone(images)
        embeddings = self.projection_head(features)
        # L2 normalize embeddings (standard for CLIP)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class CLIPModel(nn.Module):
    """
    Complete CLIP model for fine-tuning with frozen text encoder.

    Components:
    - Image encoder: ResNet50 + projection head (trainable)
    - Text encoder: Pretrained CLIP text encoder (frozen)
    """

    def __init__(self, embedding_dim=512, temperature=0.07):
        """
        Args:
            embedding_dim: Dimension of CLIP embedding space
            temperature: Temperature parameter for contrastive loss
        """
        super(CLIPModel, self).__init__()

        # Initialize image encoder with pretrained ResNet50
        self.image_encoder = CLIPImageEncoder(
            embedding_dim=embedding_dim,
            pretrained=True,
            freeze_backbone=False,  # Keep backbone trainable
        )

        # Temperature parameter for scaling logits (learnable)
        self.temperature = temperature

        self.embedding_dim = embedding_dim

    def forward(self, images, text_embeddings):
        """
        Forward pass for CLIP contrastive learning.

        Args:
            images: Tensor of shape (B, 3, 224, 224)
            text_embeddings: Precomputed text embeddings of shape (B, 512)

        Returns:
            image_embeddings: Normalized image embeddings (B, 512)
            text_embeddings: Text embeddings (already normalized, B, 512)
            logits: Similarity matrix (B, B)
        """
        # Encode images
        image_embeddings = self.image_encoder(images)  # Already normalized in encoder

        # Detach text embeddings (they're precomputed, no grads needed)
        text_embeddings = text_embeddings.detach()

        # Compute cosine similarity matrix
        logits = torch.matmul(image_embeddings, text_embeddings.T)  # (B, B)

        # Scale by temperature
        logits = logits / self.temperature

        return image_embeddings, text_embeddings, logits

    def compute_loss(self, logits):
        """
        Compute symmetric contrastive loss (InfoNCE).

        Args:
            logits: Similarity matrix of shape (B, B)

        Returns:
            loss: Scalar loss value
        """
        batch_size = logits.shape[0]

        # Ground truth: diagonal elements (matching pairs)
        labels = torch.arange(batch_size, device=logits.device)

        # Image-to-text loss
        loss_i2t = nn.functional.cross_entropy(logits, labels)

        # Text-to-image loss
        loss_t2i = nn.functional.cross_entropy(logits.T, labels)

        # Symmetric loss
        loss = (loss_i2t + loss_t2i) / 2

        return loss

    def get_trainable_params(self):
        """
        Get list of trainable parameters (image encoder + temperature).

        Returns:
            List of trainable parameters
        """
        return list(self.image_encoder.parameters())


def freeze_text_encoder(text_encoder):
    """
    Freeze all parameters in the text encoder.

    Args:
        text_encoder: CLIP text encoder model
    """
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("CLIP Model Architecture Test")
    print("=" * 60)

    # Create model
    model = CLIPModel(embedding_dim=512, temperature=0.07)

    # Print model architecture
    print("\n[Model Architecture]")
    print(f"Image Encoder: ResNet50 + Projection Head")
    print(f"Embedding Dimension: {model.embedding_dim}")
    print(f"Temperature: {model.temperature.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[Parameters]")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print(f"\n[Forward Pass Test]")
    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224)
    text_embeddings = torch.randn(batch_size, 512)

    print(f"Input shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Text embeddings: {text_embeddings.shape}")

    # Forward pass (with gradients enabled for testing)
    image_embs, text_embs, logits = model(images, text_embeddings)

    print(f"\nOutput shapes:")
    print(f"  Image embeddings: {image_embs.shape}")
    print(f"  Text embeddings: {text_embs.shape}")
    print(f"  Logits (similarity matrix): {logits.shape}")

    # Verify normalization
    img_norms = torch.norm(image_embs, p=2, dim=1)
    txt_norms = torch.norm(text_embs, p=2, dim=1)

    print(f"\n[Embedding Norms] (should be ~1.0)")
    print(
        f"  Image embedding norms: mean={img_norms.mean():.4f}, std={img_norms.std():.4f}"
    )
    print(
        f"  Text embedding norms: mean={txt_norms.mean():.4f}, std={txt_norms.std():.4f}"
    )

    # Test loss computation
    loss = model.compute_loss(logits)
    print(f"\n[Loss Test]")
    print(f"  Contrastive loss: {loss.item():.4f}")

    # Verify gradients
    print(f"\n[Gradient Check]")
    loss.backward()
    has_grad = any(p.grad is not None for p in model.get_trainable_params())
    print(f"  Gradients computed: {'✓ Yes' if has_grad else '✗ No'}")

    print("\n" + "=" * 60)
    print("✓ Model test completed successfully!")
    print("=" * 60)
