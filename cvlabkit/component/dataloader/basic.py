from torch.utils.data import DataLoader as TorchDataLoader
from cvlabkit.component.base import DataLoader
from cvlabkit.component.base import Dataset
from cvlabkit.core.config import Config


class Basic(DataLoader):
    def __init__(self, cfg: Config, dataset: Dataset, sampler=None, collate_fn=None, **kwargs):
        self.dataset = dataset

        batch_size = cfg.get("batch_size", 1)
        shuffle = cfg.get("shuffle", False)
        num_workers = cfg.get("num_workers", 0)
        pin_memory = cfg.get("pin_memory", False)

        # PyTorch DataLoader constraint: shuffle must be False if sampler is provided
        if sampler is not None:
            shuffle = False

        self.loader = TorchDataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
