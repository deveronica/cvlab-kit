from cvlabkit.component.base import Logger
import wandb

class LoggerWandb(Logger):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # wandb start (오른쪽 인자값은 기본값. config 작성된 이름을 우선적으로 사용)
        self.wandb_run = wandb.init(
            project = self.cfg.get("wandb_project", "sfda-detection-0605"),
            name = self.cfg.get("wandb_run_name", "basic-agent-run-voc-0650"),
            config = self.cfg.to_dict()
        )
        wandb.define_metric("epoch")
        wandb.define_metric("metrics/*", step_metric="epoch")

    def log_metrics(self, epoch, avg_loss, results):

        wandb.log({
                "epoch" : epoch,
                "metrics/avg_loss" : avg_loss,
                "metrics/AP@[.50:.95]": results["AP@[.50:.95]"],
                "metrics/AP50": results["AP50"],
                "metrics/AP75": results["AP75"]
        })
        

    def finalize(self):
        wandb.finish()