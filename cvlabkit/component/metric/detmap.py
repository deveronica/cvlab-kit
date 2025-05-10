from cvlabkit.component.base import Metric

class DetmapMetric(Metric):
    def __init__(self, cfg):
        self.results = []

    def update(self, model, loader):
        self.results.append(0.0)

    def compute(self):
        return {"mAP": 0.0}
