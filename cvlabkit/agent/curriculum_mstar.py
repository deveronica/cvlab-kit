"""Curriculum Learning Agent for MSTAR Classification.

Progressively introduces training samples from easy to hard based on
training-free difficulty scores computed from SAR physical indicators.

Key Features:
- Training-free difficulty estimation (no circular dependency)
- Progressive curriculum scheduling (linear, exponential, step)
- Automatic curriculum progression tracking
- Comparison with baseline (random sampling)
"""

import json
from pathlib import Path

import numpy as np

from cvlabkit.agent.basic import BasicAgent


class CurriculumMSTARAgent(BasicAgent):
    """Curriculum Learning Agent for MSTAR.

    Config requirements:
        dataloader:
            type: curriculum_mstar
            weight_strategy: 'fisher' | 'xgboost' | 'equal'
            start_ratio: 0.2  # Start with easiest 20%
            end_ratio: 1.0    # End with all samples
            schedule: 'linear' | 'exponential' | 'step'
            total_epochs: 100
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Curriculum tracking
        self.curriculum_history = []

    def train_epoch(self, epoch):
        """Train one epoch with curriculum learning."""
        # Update curriculum sampler
        if hasattr(self.dataloader, "set_epoch"):
            self.dataloader.set_epoch(epoch)

        # Get curriculum info
        curriculum_info = {}
        if hasattr(self.dataloader, "get_curriculum_info"):
            curriculum_info = self.dataloader.get_curriculum_info()

        # Log curriculum progress
        if curriculum_info:
            self.logger.info(
                f"Curriculum [Epoch {epoch}]: "
                f"{curriculum_info['n_samples']}/{curriculum_info['n_total']} samples "
                f"({curriculum_info['ratio']:.1%}), "
                f"max_difficulty={curriculum_info['difficulty_max']:.3f}"
            )

        # Standard training
        train_loader = self.dataloader.get_train_loader()
        self.model.train()

        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Logging
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        # Epoch stats
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        # Store curriculum history
        curriculum_info.update({"train_loss": avg_loss, "train_accuracy": accuracy})
        self.curriculum_history.append(curriculum_info)

        return {"loss": avg_loss, "accuracy": accuracy, "curriculum": curriculum_info}

    def save_checkpoint(self, epoch, metrics, save_path):
        """Save checkpoint with curriculum history."""
        checkpoint = super().save_checkpoint(epoch, metrics, save_path)

        # Add curriculum history
        checkpoint["curriculum_history"] = self.curriculum_history

        # Save to file
        import torch

        torch.save(checkpoint, save_path)

        # Also save curriculum history as JSON
        history_path = save_path.parent / f"{save_path.stem}_curriculum.json"
        with open(history_path, "w") as f:
            json.dump(self.curriculum_history, f, indent=2)

        self.logger.info(f"Saved curriculum history: {history_path}")

        return checkpoint

    def finalize(self):
        """Finalize training and save curriculum analysis."""
        super().finalize()

        # Generate curriculum analysis report
        if self.curriculum_history:
            self._generate_curriculum_report()

    def _generate_curriculum_report(self):
        """Generate curriculum learning analysis report."""
        import matplotlib.pyplot as plt

        output_dir = (
            Path(self.cfg.get("output_dir", "outputs"))
            / self.cfg.project
            / "curriculum_analysis"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        epochs = [h["epoch"] for h in self.curriculum_history]
        ratios = [h["ratio"] for h in self.curriculum_history]
        n_samples = [h["n_samples"] for h in self.curriculum_history]
        train_acc = [h["train_accuracy"] for h in self.curriculum_history]
        max_diff = [h["difficulty_max"] for h in self.curriculum_history]

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Curriculum progression
        ax1 = axes[0, 0]
        ax1.plot(epochs, ratios, marker="o", linewidth=2, markersize=4, color="blue")
        ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Sample Ratio", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Curriculum Progression ({self.dataloader.schedule})",
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # Plot 2: Sample count
        ax2 = axes[0, 1]
        ax2.plot(
            epochs, n_samples, marker="s", linewidth=2, markersize=4, color="green"
        )
        ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
        ax2.set_title("Training Set Size Over Time", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Training accuracy
        ax3 = axes[1, 0]
        ax3.plot(epochs, train_acc, marker="^", linewidth=2, markersize=4, color="red")
        ax3.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Training Accuracy (%)", fontsize=12, fontweight="bold")
        ax3.set_title("Training Accuracy Evolution", fontsize=14, fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Max difficulty
        ax4 = axes[1, 1]
        ax4.plot(
            epochs, max_diff, marker="D", linewidth=2, markersize=4, color="purple"
        )
        ax4.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Max Difficulty Score", fontsize=12, fontweight="bold")
        ax4.set_title(
            "Maximum Difficulty in Training Set", fontsize=14, fontweight="bold"
        )
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "curriculum_progression.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        self.logger.info(f"Curriculum analysis saved: {output_dir}")

        # Summary report
        report = []
        report.append("=" * 80)
        report.append("CURRICULUM LEARNING SUMMARY")
        report.append("=" * 80)
        report.append("")
        report.append(f"Weight Strategy: {self.dataloader.weight_strategy}")
        report.append(f"Schedule: {self.dataloader.schedule}")
        report.append(f"Start Ratio: {self.dataloader.start_ratio:.1%}")
        report.append(f"End Ratio: {self.dataloader.end_ratio:.1%}")
        report.append(f"Total Epochs: {len(epochs)}")
        report.append("")
        report.append("Training Progress:")
        report.append(f"  Initial Accuracy: {train_acc[0]:.2f}%")
        report.append(f"  Final Accuracy:   {train_acc[-1]:.2f}%")
        report.append(f"  Improvement:      {train_acc[-1] - train_acc[0]:+.2f}%")
        report.append("")
        report.append("Curriculum Progression:")
        report.append(f"  Epochs 0-25:   {np.mean(ratios[:25]):.1%} of samples")
        report.append(f"  Epochs 25-50:  {np.mean(ratios[25:50]):.1%} of samples")
        report.append(f"  Epochs 50-75:  {np.mean(ratios[50:75]):.1%} of samples")
        report.append(f"  Epochs 75-100: {np.mean(ratios[75:]):.1%} of samples")

        report_text = "\n".join(report)
        self.logger.info(f"\n{report_text}")

        with open(output_dir / "curriculum_summary.txt", "w") as f:
            f.write(report_text)
