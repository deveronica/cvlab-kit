"""
Agent for analyzing correlations between physical indicators and model predictions.

This agent trains a ResNet-18 classifier on MSTAR dataset and computes correlations
between physical image indicators (ECR, TV, Kurtosis, Shannon Entropy, GLCM Contrast)
and the model's prediction entropy.
"""

import json
import os
import torch
from tqdm import tqdm

from cvlabkit.core.agent import Agent
from cvlabkit.core.config import Config
from cvlabkit.core.creator import ComponentCreator


class MSTARCorrelation(Agent):
    """
    Agent for training ResNet-18 on MSTAR and analyzing physical indicator correlations.

    This agent extends the standard classification workflow to:
    1. Train a ResNet-18 classifier on MSTAR dataset
    2. Compute physical indicators (ECR, TV, Kurtosis, Entropy, GLCM) for test images
    3. Compute prediction entropy for each test sample
    4. Analyze correlations between physical indicators and prediction entropy
    5. Save detailed results for further analysis
    """

    def setup(self):
        """
        Creates and sets up all components required for correlation analysis.

        Sets up:
        - Model (ResNet-18)
        - Optimizer
        - Loss function
        - Train/Test dataloaders
        - Physical indicators metric
        - Standard accuracy metric
        """
        print("Initializing MSTAR Correlation Analysis components...")

        # Initialize early stopping flag
        self._stop_training = False

        self.device = (
            f"cuda:{self.cfg.get('device', 0)}" if torch.cuda.is_available() else "cpu"
        )

        # Create model
        self.model = self.create.model().to(self.device)
        print(f"Model: {self.cfg.model}")

        # Create optimizer
        self.optimizer = self.create.optimizer(self.model.parameters())

        # Create loss function
        self.loss_fn = self.create.loss()

        # Create transforms
        transform = self.create.transform() if "transform" in self.cfg else None

        # Create datasets
        train_dataset = self.create.dataset.train(transform=transform)
        val_dataset = self.create.dataset.val(transform=transform)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        # Create dataloaders
        self.train_loader = self.create.dataloader.train(train_dataset)
        self.val_loader = self.create.dataloader.val(val_dataset)

        # Create metrics
        # Physical indicators metric for correlation analysis
        self.physical_metric = self.create.metric.physical()

        # Standard accuracy metric for monitoring training
        self.accuracy_metric = self.create.metric.accuracy()

        print(f"Components initialized. Using device: {self.device}")

    def train(self):
        """
        Override train method to support early stopping based on target accuracy.
        """
        for epoch in range(self.cfg.epochs):
            self.current_epoch = epoch

            # Training phase
            self.model.train()
            print(f"\nStarting epoch {epoch + 1}/{self.cfg.epochs}...")

            # Train for one epoch
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training"):
                self.train_step(batch)

            # Evaluation phase
            self.model.eval()
            with torch.no_grad():
                self.evaluate()

            # Check early stopping
            if self._stop_training:
                print(f"\n✓ Training stopped early at epoch {epoch + 1}")
                break

    def train_step(self, batch):
        """
        Performs a single training step.

        Args:
            batch: Tuple of (inputs, labels)

        Returns:
            Dictionary with training loss
        """
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate_step(self, batch):
        """
        Performs a single validation step with physical indicator computation.

        This method differs from standard validation by:
        1. Computing physical indicators from input images
        2. Computing prediction entropy from model outputs
        3. Updating both accuracy and physical indicators metrics

        Args:
            batch: Tuple of (inputs, labels)

        Returns:
            Dictionary with validation loss
        """
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)

        # Update metrics
        # Physical indicators metric needs images, predictions, and targets
        self.physical_metric.update(images=inputs, preds=outputs, targets=labels)

        # Standard accuracy metric
        self.accuracy_metric.update(preds=outputs, targets=labels)

        return {"loss": loss.item()}

    def evaluate(self):
        """
        Evaluates the model on validation set and computes correlation analysis.

        This method:
        1. Runs validation on validation set
        2. Computes all physical indicators and prediction entropies
        3. Calculates correlations
        4. Prints summary statistics
        5. Saves detailed results to JSON file
        """
        print("\n" + "="*80)
        print(f"Evaluating at Epoch {self.current_epoch + 1}/{self.cfg.epochs}")
        print("="*80)

        # Reset metrics
        self.physical_metric.reset()
        self.accuracy_metric.reset()

        # Run validation loop
        for batch in self.val_loader:
            self.validate_step(batch)

        # Compute results
        physical_results = self.physical_metric.compute()
        accuracy_results = self.accuracy_metric.compute()

        # Print summary
        print("\n" + "-"*80)
        print("EVALUATION SUMMARY")
        print("-"*80)
        print(f"Validation Accuracy: {accuracy_results['accuracy']:.4f}")
        print(f"Samples Processed: {physical_results['num_samples']}")

        print("\n" + "-"*80)
        print("PHYSICAL INDICATORS STATISTICS")
        print("-"*80)
        print(f"ECR (Energy Compaction):     Mean={physical_results['ecr_mean']:.4f}, Std={physical_results['ecr_std']:.4f}")
        print(f"TV (Total Variation):        Mean={physical_results['tv_mean']:.4f}, Std={physical_results['tv_std']:.4f}")
        print(f"Kurtosis:                    Mean={physical_results['kurtosis_mean']:.4f}, Std={physical_results['kurtosis_std']:.4f}")
        print(f"Shannon Entropy:             Mean={physical_results['entropy_mean']:.4f}, Std={physical_results['entropy_std']:.4f}")
        print(f"GLCM Contrast:               Mean={physical_results['glcm_contrast_mean']:.4f}, Std={physical_results['glcm_contrast_std']:.4f}")
        print(f"GLCM Entropy:                Mean={physical_results['glcm_entropy_mean']:.4f}, Std={physical_results['glcm_entropy_std']:.4f}")
        print(f"ENL:                         Mean={physical_results['enl_mean']:.4f}, Std={physical_results['enl_std']:.4f}")

        print("\n" + "-"*80)
        print("MODEL UNCERTAINTY STATISTICS")
        print("-"*80)
        print(f"Prediction Entropy:          Mean={physical_results['pred_entropy_mean']:.4f}, Std={physical_results['pred_entropy_std']:.4f}")
        print(f"Cross Entropy Loss:          Mean={physical_results['ce_loss_mean']:.4f}, Std={physical_results['ce_loss_std']:.4f}")
        print(f"Pred Entropy ↔ CE Loss:      ρ = {physical_results['corr_pred_entropy_ce_loss']:+.4f}")

        print("\n" + "="*80)
        print("CORRELATION ANALYSIS: Physical Indicators ↔ Prediction Entropy")
        print("="*80)
        print(f"ECR ↔ Prediction Entropy:            ρ = {physical_results['corr_ecr_pred_entropy']:+.4f}")
        print(f"TV ↔ Prediction Entropy:             ρ = {physical_results['corr_tv_pred_entropy']:+.4f}")
        print(f"Kurtosis ↔ Prediction Entropy:       ρ = {physical_results['corr_kurtosis_pred_entropy']:+.4f}")
        print(f"Shannon Entropy ↔ Prediction Entropy: ρ = {physical_results['corr_entropy_pred_entropy']:+.4f}")
        print(f"GLCM Contrast ↔ Prediction Entropy:  ρ = {physical_results['corr_glcm_contrast_pred_entropy']:+.4f}")
        print(f"GLCM Entropy ↔ Prediction Entropy:   ρ = {physical_results['corr_glcm_entropy_pred_entropy']:+.4f}")
        print(f"ENL ↔ Prediction Entropy:            ρ = {physical_results['corr_enl_pred_entropy']:+.4f}")

        print("\n" + "="*80)
        print("CORRELATION ANALYSIS: Physical Indicators ↔ Cross Entropy Loss")
        print("="*80)
        print(f"ECR ↔ CE Loss:                       ρ = {physical_results['corr_ecr_ce_loss']:+.4f}")
        print(f"TV ↔ CE Loss:                        ρ = {physical_results['corr_tv_ce_loss']:+.4f}")
        print(f"Kurtosis ↔ CE Loss:                  ρ = {physical_results['corr_kurtosis_ce_loss']:+.4f}")
        print(f"Shannon Entropy ↔ CE Loss:           ρ = {physical_results['corr_entropy_ce_loss']:+.4f}")
        print(f"GLCM Contrast ↔ CE Loss:             ρ = {physical_results['corr_glcm_contrast_ce_loss']:+.4f}")
        print(f"GLCM Entropy ↔ CE Loss:              ρ = {physical_results['corr_glcm_entropy_ce_loss']:+.4f}")
        print(f"ENL ↔ CE Loss:                       ρ = {physical_results['corr_enl_ce_loss']:+.4f}")
        print("="*80 + "\n")

        # Save detailed results to file
        self._save_correlation_results(physical_results, accuracy_results)

        # Save human-readable summary
        self._save_correlation_summary(physical_results, accuracy_results)

        # Save per-image CSV table
        self._save_per_image_table(physical_results)

        # Save checkpoint periodically
        save_freq = self.cfg.get("save_freq", None)
        if save_freq is not None and (self.current_epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(
                self.cfg.get("output_dir", "./outputs/mstar_correlation"),
                f"checkpoint_epoch_{self.current_epoch:03d}.pt"
            )
            self.save(checkpoint_path)

        # Check if target accuracy reached
        target_accuracy = self.cfg.get("target_accuracy", None)
        if target_accuracy is not None and accuracy_results['accuracy'] >= target_accuracy:
            print("\n" + "="*80)
            print(f"🎯 TARGET ACCURACY REACHED: {accuracy_results['accuracy']:.4f} >= {target_accuracy:.4f}")
            print(f"Stopping training at epoch {self.current_epoch + 1}")
            print("="*80 + "\n")
            # Signal to stop training
            self._stop_training = True

    def _save_correlation_results(self, physical_results: dict, accuracy_results: dict):
        """
        Saves detailed correlation results to JSON file.

        Args:
            physical_results: Results from PhysicalIndicators metric
            accuracy_results: Results from Accuracy metric
        """
        # Prepare results dictionary (correlation first for easy access)
        results = {
            "experiment": {
                "epoch": self.current_epoch,
                "model": self.cfg.model,
                "dataset": "MSTAR",
                "device": self.device,
            },
            "performance": {
                "accuracy": accuracy_results['accuracy'],
                "num_samples": physical_results['num_samples'],
            },
            "correlations": {
                "pred_entropy_ce_loss": physical_results['corr_pred_entropy_ce_loss'],
                "ecr_pred_entropy": physical_results['corr_ecr_pred_entropy'],
                "tv_pred_entropy": physical_results['corr_tv_pred_entropy'],
                "kurtosis_pred_entropy": physical_results['corr_kurtosis_pred_entropy'],
                "entropy_pred_entropy": physical_results['corr_entropy_pred_entropy'],
                "glcm_contrast_pred_entropy": physical_results['corr_glcm_contrast_pred_entropy'],
                "glcm_entropy_pred_entropy": physical_results['corr_glcm_entropy_pred_entropy'],
                "enl_pred_entropy": physical_results['corr_enl_pred_entropy'],
                "ecr_ce_loss": physical_results['corr_ecr_ce_loss'],
                "tv_ce_loss": physical_results['corr_tv_ce_loss'],
                "kurtosis_ce_loss": physical_results['corr_kurtosis_ce_loss'],
                "entropy_ce_loss": physical_results['corr_entropy_ce_loss'],
                "glcm_contrast_ce_loss": physical_results['corr_glcm_contrast_ce_loss'],
                "glcm_entropy_ce_loss": physical_results['corr_glcm_entropy_ce_loss'],
                "enl_ce_loss": physical_results['corr_enl_ce_loss'],
            },
            "physical_indicators_summary": {
                "ecr": {
                    "mean": physical_results['ecr_mean'],
                    "std": physical_results['ecr_std'],
                },
                "tv": {
                    "mean": physical_results['tv_mean'],
                    "std": physical_results['tv_std'],
                },
                "kurtosis": {
                    "mean": physical_results['kurtosis_mean'],
                    "std": physical_results['kurtosis_std'],
                },
                "shannon_entropy": {
                    "mean": physical_results['entropy_mean'],
                    "std": physical_results['entropy_std'],
                },
                "glcm_contrast": {
                    "mean": physical_results['glcm_contrast_mean'],
                    "std": physical_results['glcm_contrast_std'],
                },
                "glcm_entropy": {
                    "mean": physical_results['glcm_entropy_mean'],
                    "std": physical_results['glcm_entropy_std'],
                },
                "enl": {
                    "mean": physical_results['enl_mean'],
                    "std": physical_results['enl_std'],
                },
                "prediction_entropy": {
                    "mean": physical_results['pred_entropy_mean'],
                    "std": physical_results['pred_entropy_std'],
                },
                "cross_entropy_loss": {
                    "mean": physical_results['ce_loss_mean'],
                    "std": physical_results['ce_loss_std'],
                },
            },
            "raw_data": {
                "physical_indicators": {
                    "ecr": physical_results['ecr'],
                    "tv": physical_results['tv'],
                    "kurtosis": physical_results['kurtosis'],
                    "shannon_entropy": physical_results['shannon_entropy'],
                    "glcm_contrast": physical_results['glcm_contrast'],
                    "glcm_entropy": physical_results['glcm_entropy'],
                    "enl": physical_results['enl'],
                },
                "prediction_entropy": physical_results['prediction_entropy'],
                "cross_entropy_loss": physical_results['cross_entropy_loss'],
                "targets": [int(t) for t in physical_results['targets']],
                "correct_predictions": physical_results['correct'],
            },
        }

        # Add optional logits and indices
        if 'logits' in physical_results:
            results['raw_data']['logits'] = physical_results['logits']

        if 'indices' in physical_results:
            results['raw_data']['indices'] = physical_results['indices']

        # Save to JSON file
        output_dir = self.cfg.get("output_dir", "./outputs/mstar_correlation")
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(
            output_dir,
            f"correlation_epoch_{self.current_epoch:03d}.json"
        )

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Detailed results saved to: {output_file}\n")

    def _save_correlation_summary(self, physical_results: dict, accuracy_results: dict):
        """
        Saves human-readable correlation summary to text file.

        Args:
            physical_results: Results from PhysicalIndicators metric
            accuracy_results: Results from Accuracy metric
        """
        output_dir = self.cfg.get("output_dir", "./outputs/mstar_correlation")
        os.makedirs(output_dir, exist_ok=True)

        summary_file = os.path.join(
            output_dir,
            f"summary_epoch_{self.current_epoch:03d}.txt"
        )

        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MSTAR PHYSICAL INDICATORS CORRELATION ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")

            # Experiment info
            f.write(f"Epoch: {self.current_epoch + 1}/{self.cfg.epochs}\n")
            f.write(f"Model: {self.cfg.model}\n")
            f.write(f"Dataset: MSTAR\n")
            f.write(f"Device: {self.device}\n\n")

            # Performance
            f.write("-"*80 + "\n")
            f.write("MODEL PERFORMANCE\n")
            f.write("-"*80 + "\n")
            f.write(f"Validation Accuracy:  {accuracy_results['accuracy']:.4f} ({accuracy_results['accuracy']*100:.2f}%)\n")
            f.write(f"Samples Processed:    {physical_results['num_samples']}\n\n")

            # Physical indicators statistics
            f.write("-"*80 + "\n")
            f.write("PHYSICAL INDICATORS STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Indicator':<25s} {'Mean':>12s} {'Std':>12s}\n")
            f.write("-"*80 + "\n")
            f.write(f"{'ECR (Energy Compaction)':<25s} {physical_results['ecr_mean']:>12.4f} {physical_results['ecr_std']:>12.4f}\n")
            f.write(f"{'TV (Total Variation)':<25s} {physical_results['tv_mean']:>12.2f} {physical_results['tv_std']:>12.2f}\n")
            f.write(f"{'Kurtosis':<25s} {physical_results['kurtosis_mean']:>12.4f} {physical_results['kurtosis_std']:>12.4f}\n")
            f.write(f"{'Shannon Entropy':<25s} {physical_results['entropy_mean']:>12.4f} {physical_results['entropy_std']:>12.4f}\n")
            f.write(f"{'GLCM Contrast':<25s} {physical_results['glcm_contrast_mean']:>12.2f} {physical_results['glcm_contrast_std']:>12.2f}\n")
            f.write(f"{'GLCM Entropy':<25s} {physical_results['glcm_entropy_mean']:>12.4f} {physical_results['glcm_entropy_std']:>12.4f}\n")
            f.write(f"{'ENL':<25s} {physical_results['enl_mean']:>12.4f} {physical_results['enl_std']:>12.4f}\n")
            f.write(f"{'Prediction Entropy':<25s} {physical_results['pred_entropy_mean']:>12.4f} {physical_results['pred_entropy_std']:>12.4f}\n\n")

            # Correlations (main focus)
            f.write("="*80 + "\n")
            f.write("CORRELATION ANALYSIS (Spearman's ρ)\n")
            f.write("="*80 + "\n")
            f.write(f"{'Physical Indicator':<35s} {'Spearman ρ':>15s} {'Interpretation':>20s}\n")
            f.write("-"*80 + "\n")

            correlations = [
                ("ECR ↔ Prediction Entropy", physical_results['corr_ecr_pred_entropy']),
                ("TV ↔ Prediction Entropy", physical_results['corr_tv_pred_entropy']),
                ("Kurtosis ↔ Prediction Entropy", physical_results['corr_kurtosis_pred_entropy']),
                ("Shannon Entropy ↔ Pred. Entropy", physical_results['corr_entropy_pred_entropy']),
                ("GLCM Contrast ↔ Pred. Entropy", physical_results['corr_glcm_contrast_pred_entropy']),
                ("GLCM Entropy ↔ Pred. Entropy", physical_results['corr_glcm_entropy_pred_entropy']),
                ("ENL ↔ Pred. Entropy", physical_results['corr_enl_pred_entropy']),
            ]

            # Sort by absolute correlation strength
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            for name, corr in correlations:
                interpretation = self._interpret_correlation(corr)
                f.write(f"{name:<35s} {corr:>+15.4f} {interpretation:>20s}\n")

            f.write("="*80 + "\n\n")

            # Interpretation guide
            f.write("INTERPRETATION GUIDE\n")
            f.write("-"*80 + "\n")
            f.write("Spearman's ρ measures monotonic relationship between variables:\n")
            f.write("  • |ρ| < 0.1:  Negligible correlation\n")
            f.write("  • |ρ| < 0.3:  Weak correlation\n")
            f.write("  • |ρ| < 0.5:  Moderate correlation\n")
            f.write("  • |ρ| < 0.7:  Strong correlation\n")
            f.write("  • |ρ| ≥ 0.7:  Very strong correlation\n\n")
            f.write("Positive ρ: As physical indicator increases, prediction entropy increases\n")
            f.write("            (More of this property → Model more uncertain)\n\n")
            f.write("Negative ρ: As physical indicator increases, prediction entropy decreases\n")
            f.write("            (More of this property → Model more confident)\n\n")

            f.write("="*80 + "\n")

        print(f"Human-readable summary saved to: {summary_file}\n")

    def _save_per_image_table(self, physical_results: dict):
        """
        Saves per-image table with all indicators to CSV file.

        CSV columns: image_id, class, prediction_entropy, ecr, tv, kurtosis,
                     shannon_entropy, glcm_contrast, glcm_entropy, enl, correct

        Args:
            physical_results: Results from PhysicalIndicators metric
        """
        import csv

        output_dir = self.cfg.get("output_dir", "./outputs/mstar_correlation")
        os.makedirs(output_dir, exist_ok=True)

        csv_file = os.path.join(
            output_dir,
            f"per_image_epoch_{self.current_epoch:03d}.csv"
        )

        # Prepare data
        num_samples = physical_results['num_samples']

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'image_id',
                'class',
                'prediction_entropy',
                'cross_entropy_loss',
                'ecr',
                'tv',
                'kurtosis',
                'shannon_entropy',
                'glcm_contrast',
                'glcm_entropy',
                'enl',
                'correct'
            ])

            # Write data rows
            for i in range(num_samples):
                writer.writerow([
                    i,  # image_id
                    int(physical_results['targets'][i]),  # class
                    physical_results['prediction_entropy'][i],  # prediction_entropy
                    physical_results['cross_entropy_loss'][i],  # cross_entropy_loss
                    physical_results['ecr'][i],  # ecr
                    physical_results['tv'][i],  # tv
                    physical_results['kurtosis'][i],  # kurtosis
                    physical_results['shannon_entropy'][i],  # shannon_entropy
                    physical_results['glcm_contrast'][i],  # glcm_contrast
                    physical_results['glcm_entropy'][i],  # glcm_entropy
                    physical_results['enl'][i],  # enl
                    int(physical_results['correct'][i])  # correct
                ])

        print(f"Per-image CSV table saved to: {csv_file}\n")

    def _interpret_correlation(self, rho: float) -> str:
        """Interpret correlation strength."""
        abs_rho = abs(rho)
        if abs_rho < 0.1:
            return "Negligible"
        elif abs_rho < 0.3:
            return "Weak"
        elif abs_rho < 0.5:
            return "Moderate"
        elif abs_rho < 0.7:
            return "Strong"
        else:
            return "Very Strong"

    def save(self, path: str):
        """Saves model checkpoint."""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load(self, path: str):
        """Loads model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {path}")
