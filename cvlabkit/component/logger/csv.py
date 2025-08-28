import csv
import os
import re
from typing import Dict, Any

from cvlabkit.component.base import Logger
from cvlabkit.core.config import Config


class Csv(Logger):
    """
    A logger that saves experiment metrics to a CSV file.

    Upon the first call to `log_metrics`, it creates a CSV file and writes a
    header row. Subsequent calls append new rows for each step.
    """
    def __init__(self, cfg: Config):
        """
        Initializes the CSVLogger.

        Args:
            cfg (Config): The configuration object. Expected keys:
                - "log_dir" (str, optional): Directory to save log files. Defaults to "./logs".
                - "run_name" (str, optional): A name for the run, used as the CSV filename.
                                              Supports placeholders like {{key}}.
                                              Defaults to "experiment".
        """
        log_dir = cfg.get("log_dir", "./logs")
        run_name = cfg.get("run_name", "experiment")
        
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{run_name}.csv")
        
        self.file = open(self.log_path, 'w', newline='')
        self.writer = None
        
        # Save the config alongside the log file for reproducibility
        self.log_hyperparams(cfg.to_dict())

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Logs a dictionary of metrics for a given step.

        Args:
            metrics (Dict[str, float]): A dictionary of metric names and their values.
            step (int): The current step or epoch number.
        """
        if not metrics:
            return

        # Prepare a flat dictionary including the step
        log_row = {'step': step, **metrics}

        # On the first call, create the writer and write the header
        if self.writer is None:
            self.fieldnames = log_row.keys()
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        
        self.writer.writerow(log_row)
        self.file.flush() # Ensure data is written to disk immediately

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Saves the hyperparameter configuration to a YAML file."""
        import yaml
        config_path = self.log_path.replace(".csv", "_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(params, f, default_flow_style=False)

    def finalize(self) -> None:
        """Closes the CSV file."""
        if self.file:
            self.file.close()