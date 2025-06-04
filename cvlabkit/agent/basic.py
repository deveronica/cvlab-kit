import torch
from torch.utils.data import DataLoader
from cvlabkit.core.agent import Agent as BaseAgent
from cvlabkit.core.creator import Creator
import wandb

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

        # wandb start (오른쪽 인자값은 기본값. config 작성된 이름을 우선적으로 사용)
        self.wandb_run = wandb.init(
            project = self.cfg.get("wandb_project", "sfda-detection"),
            name = self.cfg.get("wandb_run_name", "basic-agent-run-voc"),
            config = self.cfg.to_dict()
        )
        wandb.define_metric("epoch")
        wandb.define_metric("metrics/*", step_metric="epoch")

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
                
                ## wandb setp log add
                wandb.log({
                    "train/loss_step": loss.item(),
                })

            avg_loss = total_loss / num_batches
            print(f"[fit] Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}")
            
            results = self.evaluate()
            print(f"[fit] Epoch {epoch + 1}: AP@[.50:.95] = {results['AP@[.50:.95]']:.4f}, AP50 = {results['AP50']:.4f}, AP75 = {results['AP75']:.4f}")
            
                        # wandb epoch log add
            wandb.log({
                "epoch" : epoch,
                "metrics/avg_loss" : avg_loss,
                "metrics/AP@[.50:.95]": results["AP@[.50:.95]"],
                "metrics/AP50": results["AP50"],
                "metrics/AP75": results["AP75"]
            })
        
        # wandb finish
        wandb.finish()

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
