import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch


class COCODatasetYOLO(Dataset):
    def __init__(self, root_dir, split="train2014", transform=None):
        """
        Args:
            root_dir (str): Path to coco2014 folder.
            split (str): One of ['train2014', 'val2014', 'test2014'].
            transform: Torchvision transforms for images.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.img_dir = os.path.join(root_dir, "images", split)
        self.label_dir = os.path.join(
            root_dir, "labels", split.replace("test", "val")
        )  # no test labels

        # List images
        self.images = sorted(
            [
                f
                for f in os.listdir(self.img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

    def __len__(self):
        return len(self.images)

    def load_labels(self, label_path):
        """
        YOLO label format:
        class_id x_center y_center w h
        """
        boxes = []
        if not os.path.exists(label_path):
            return torch.zeros((0, 5), dtype=torch.float32)

        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.strip().split())
                boxes.append([cls, x, y, w, h])

        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Load label (text file with same name but .txt)
        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)
        labels = self.load_labels(label_path)

        if self.transform:
            img = self.transform(img)

        return img, labels


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torchvision import transforms

    root = r"./data/coco2014"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = COCODatasetYOLO(root, split="train2014", transform=transform)

    label_text = []
    with open(os.path.join(root, "coco.names"), "r") as f:
        file = f.readlines()
        file = [str(f).replace("\n", "") for f in file]
        label_text = file

    # pick a random sample
    idx = random.randint(0, len(dataset) - 1)
    img, labels = dataset[idx]

    # convert tensor → numpy HWC for plotting
    img_np = img.permute(1, 2, 0).numpy()

    h, w = img_np.shape[:2]

    fig, ax = plt.subplots(1)
    ax.imshow(img_np)

    # Draw YOLO boxes
    for lbl in labels:
        cls, xc, yc, bw, bh = lbl

        # Convert YOLO -> pixel xywh
        box_w = bw * w
        box_h = bh * h
        box_x = (xc * w) - box_w / 2
        box_y = (yc * h) - box_h / 2

        rect = patches.Rectangle(
            (box_x, box_y),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            box_x,
            box_y - 3,
            f"class {label_text[int(cls)]}",
            color="yellow",
            fontsize=10,
            backgroundcolor="black",
        )

    plt.title(f"Sample #{idx} from train2014")
    plt.axis("off")
    plt.show()
