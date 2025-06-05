import torch
from torch.utils.data import DataLoader
from cvlabkit.core.agent import Agent as BaseAgent
from cvlabkit.core.creator import Creator


def collate_fn(batch):
    return [b["img"] for b in batch], [b["target"] for b in batch]


class Agent(BaseAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        create = Creator(cfg)
        self.dataset = create.dataset()
        self.model = create.model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def dry_run(self):
        loader = DataLoader(self.dataset, batch_size=1, collate_fn=collate_fn)
        x, y = next(iter(loader))
        x = [i.to(self.device) for i in x]
        y = [
            {k: v.to(self.device) if hasattr(v, "to") else v for k, v in t.items()}
            for t in y
        ]
        self.model.train()
        losses = self.model(x, y)
        print("[dry_run] loss keys:", list(losses.keys()))

    def fit(self, epochs=10):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)

        for epoch in range(epochs):
            self.model.train()
            
            total_loss = 0.0
            num_batches = 0

            for x, y in loader:
                x = [i.to(self.device) for i in x]
                y = [
                    {k: v.to(self.device) if hasattr(v, "to") else v for k, v in t.items()}
                    for t in y
                ]
                losses = self.model(x, y)
                loss = sum(losses.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"[fit] Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}")
            
            results = self.evaluate()
            print(f"[fit] Epoch {epoch + 1}: AP@[.50:.95] = {results['AP@[.50:.95]']:.4f}, AP50 = {results['AP50']:.4f}, AP75 = {results['AP75']:.4f}")
            
    def evaluate(self):
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=1, collate_fn=collate_fn)
        
        metric = Creator(self.cfg).metric()

        with torch.no_grad():
            for x, y in loader:
                x = [i.to(self.device) for i in x]
                y = [
                    {k: v.to(self.device) if hasattr(v, "to") else v for k, v in t.items()}
                    for t in y
                ]
                outputs = self.model(x)

                for img_tensor, gt, pred in zip(x, y, outputs):
                    image_id = gt["image_id"].item() if hasattr(gt["image_id"], "item") else gt["image_id"]
                    boxes = pred["boxes"].detach().cpu().numpy()
                    scores = pred["scores"].detach().cpu().numpy()
                    labels = pred["labels"]
                    labels = labels.tolist() if hasattr(labels, "tolist") else labels

                    label_names = [self.cfg.class_names[idx - 1] for idx in labels]

                    metric.update(
                        image_id=image_id,
                        boxes=boxes,
                        scores=scores,
                        labels=label_names,
                    )

        return metric.compute()
