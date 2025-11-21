import os
from typing import Any, Dict

import pandas as pd

from cvlabkit.component.base import Logger
from cvlabkit.core.config import Config


class Csv(Logger):
    """A logger that saves experiment metrics to a CSV file using pandas.
    This version handles dynamic addition of new metrics during the run.
    """

    def __init__(self, cfg: Config):
        """Initializes the CSVLogger.

        Args:
            cfg (Config): The configuration object. Expected keys:
                - "log_dir" (str, optional): Directory to save log files. Defaults to "./logs".
                - "run_name" (str, optional): A name for the run, used as the CSV filename.
                                              Defaults to "experiment".
                - "run_uid" (str, optional): Unique run identifier. If provided, takes precedence over run_name.
        """
        log_dir = cfg.get("log_dir", "./logs")
        # Prefer run_uid over run_name for web_helper compatibility
        run_name = cfg.get("run_uid") or cfg.get("run_name", "experiment")

        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{run_name}.csv")

        if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0:
            self.df = pd.read_csv(self.log_path, index_col="step")
        else:
            self.df = pd.DataFrame()

        # Save the config alongside the log file for reproducibility
        self.log_hyperparams(cfg.to_dict())

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Logs a dictionary of metrics for a given step.
        Updates the CSV file, adding new columns if necessary.

        Args:
            metrics (Dict[str, float]): A dictionary of metric names and their values.
            step (int): The current step or epoch number.
        """
        if not metrics:
            return

        for key, value in metrics.items():
            self.df.loc[step, key] = value

        self.df.sort_index(inplace=True)
        self.df.to_csv(self.log_path, index_label="step")

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Saves the hyperparameter configuration to a YAML file."""
        import yaml

        config_path = self.log_path.replace(".csv", "_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False)

    def finalize(self) -> None:
        """Saves the final DataFrame to the CSV file."""
        self.df.to_csv(self.log_path, index_label="step")
