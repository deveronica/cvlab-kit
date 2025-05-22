import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
import torch
from torchvision.transforms import functional as F

from torch.utils.data import Dataset


CLASS_NAMES = (
    "aeroplane", "bird", "boat", "bottle", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "pottedplant", "sheep", "sofa", "tvmonitor", 
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
)


class VOCDetectionDataset(Dataset):
    def __init__(self, cfg):
        root = Path(cfg.root)
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        self.root = root

        self.split = getattr(cfg, "split", "train")
        self.class_names = getattr(cfg, "class_names", CLASS_NAMES)

        with open(self.root / "ImageSets" / "Main" / f"{self.split}.txt") as f:
            self.file_ids = [line.strip() for line in f]

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_id = self.file_ids[idx]
        anno_path = self.root / "Annotations" / f"{file_id}.xml"
        image_path = self.root / "JPEGImages" / f"{file_id}.jpg"

        image = Image.open(image_path).convert("RGB")
        image = F.to_tensor(image)
        tree = ET.parse(str(anno_path))
        height = int(tree.find("./size/height").text)
        width = int(tree.find("./size/width").text)

        boxes = []
        labels = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text) - 1
            ymin = float(bbox.find("ymin").text) - 1
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_names.index(cls))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return {
            "img": image,
            "target": {
                "boxes": boxes,
                "labels": labels,
                "image_id": idx,
                "size": (height, width)
            }
        }

