import torch
from cvlabkit.core.agent import Agent
from cvlabkit.core.creator import Creator

class FixMatchAgent(Agent):
    def __init__(self, cfg):
        super().__init__(cfg)
        create = Creator(cfg)
        self.optimizer = create.optimizer()
        self.dataloader = create.dataloader()
        self.loss_fn = create.loss()
        self.confidence_threshold = cfg.get("confidence_threshold", 0.95)
        self.pretrain_source = cfg.get("pretrain_source")

    def train_step(self, batch):
        if self.pretrain_source:
            image, target = batch
            outputs = self.model(image, target)
            loss = sum(v for v in outputs.values())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.detach()
        
        else:
        images, targets = batch

        # Supervised loss
        outputs = self.model(images, targets)
        loss = sum(v for v in outputs.values())

        # Unlabeled: weak/strong augmentation
        weak_imgs = [self.weak_augment(img) for img in images]
        strong_imgs = [self.strong_augment(img) for img in images]

        with torch.no_grad():
            pseudo_outputs = self.model(weak_imgs)
        pseudo_labels = self.get_pseudo_labels(pseudo_outputs)

        if pseudo_labels:
            strong_outputs = self.model(strong_imgs, pseudo_labels)
            unsup_loss = sum(v for v in strong_outputs.values())
            loss += unsup_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def weak_augment(self, img):
        # 약한 증강 (예: 플립)
        return img

    def strong_augment(self, img):
        # 강한 증강 (예: RandAugment 등)
        return img

    def get_pseudo_labels(self, outputs):
        # confidence threshold 적용
        # outputs: List[Dict] (torchvision detection output)
        pseudo_labels = []
        for out in outputs:
            scores = out['scores']
            keep = scores > self.confidence_threshold
            if keep.sum() == 0:
                pseudo_labels.append(None)
            else:
                pseudo_labels.append({
                    'boxes': out['boxes'][keep],
                    'labels': out['labels'][keep]
                })
        return pseudo_labels

    def fit(self):
        for epoch in range(1, 1 + self.cfg.get("epochs", 10)):
            for batch, unlabeled_batch in zip(self.dataloader, self.unlabeled_loader):
                loss = self.train_step(batch, unlabeled_batch)
                print(f"Epoch {epoch}, Loss: {loss}")