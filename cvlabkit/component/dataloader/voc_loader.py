from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from component.dataset.voc import VOCDetectionDataset


def collate_fn(batch):
    return {
        "img": [ToTensor()(b["img"]) for b in batch],
        "target": [b["target"] for b in batch]
    }


def create(cfg):
    dataset = VOCDetectionDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=cfg.get("shuffle", True),
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_fn
    )
