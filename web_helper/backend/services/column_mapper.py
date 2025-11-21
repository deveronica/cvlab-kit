"""Column Mapping Service

Implements smart column mapping algorithms for parameter name unification.

Algorithms:
1. Fuzzy String Matching: SequenceMatcher for similar names
2. Semantic Matching: Synonym groups and common abbreviations
3. Value Range Matching: Overlap in parameter value distributions
4. Distribution Shape Matching: Statistical similarity using KS test
5. Context Analysis: YAML config structure and naming patterns

Confidence Score Weights:
- Fuzzy: 0.25
- Semantic: 0.30
- Value Range: 0.20
- Distribution: 0.15
- Context: 0.10
"""

from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from scipy import stats
from sqlalchemy.orm import Session

from ..models.column_mapping import ColumnMapping, ColumnSuggestion

# Semantic synonym groups for common ML parameters
SEMANTIC_GROUPS = {
    "learning_rate": {"lr", "learning_rate", "learn_rate", "lrate", "alpha"},
    "batch_size": {"batch_size", "batch", "bs", "batchsize"},
    "epochs": {"epochs", "epoch", "n_epochs", "num_epochs", "max_epochs"},
    "momentum": {"momentum", "mom", "beta1"},
    "weight_decay": {"weight_decay", "wd", "decay", "l2", "regularization"},
    "dropout": {"dropout", "drop", "dropout_rate", "drop_rate", "p_dropout"},
    "num_layers": {"num_layers", "n_layers", "depth", "layers"},
    "hidden_size": {"hidden_size", "hidden_dim", "h_dim", "hidden_units"},
    "seed": {"seed", "random_seed", "rand_seed", "rng_seed"},
    "optimizer": {"optimizer", "optim", "opt"},
    "loss": {"loss", "loss_fn", "loss_function", "criterion"},
    "accuracy": {"accuracy", "acc", "train_acc", "val_acc", "test_acc"},
    "precision": {"precision", "prec"},
    "recall": {"recall", "rec"},
    "f1_score": {"f1", "f1_score", "f1score", "fscore"},
}


class ColumnMapperService:
    """Service for intelligent column mapping and suggestion generation"""

    def __init__(self, db: Session):
        self.db = db

    # ========================================================================
    # Algorithm 1: Fuzzy String Matching
    # ========================================================================

    def fuzzy_similarity(self, source: str, target: str) -> float:
        """Calculate fuzzy string similarity using SequenceMatcher.

        Returns similarity score between 0.0 and 1.0.
        """
        return SequenceMatcher(None, source.lower(), target.lower()).ratio()

    # ========================================================================
    # Algorithm 2: Semantic Matching
    # ========================================================================

    def semantic_similarity(
        self, source: str, target: str
    ) -> Tuple[float, Optional[str]]:
        """Check if two column names belong to the same semantic group.

        Returns:
            (score, group_name): score is 1.0 if same group, 0.0 otherwise
        """
        source_lower = source.lower()
        target_lower = target.lower()

        for group_name, synonyms in SEMANTIC_GROUPS.items():
            if source_lower in synonyms and target_lower in synonyms:
                return 1.0, group_name

        return 0.0, None

    # ========================================================================
    # Algorithm 3: Value Range Matching
    # ========================================================================

    def value_range_similarity(
        self, source_values: List[float], target_values: List[float]
    ) -> float:
        """Calculate overlap between value ranges of two parameters.

        Uses Jaccard similarity of value ranges:
        overlap = intersection / union of [min, max] ranges
        """
        if not source_values or not target_values:
            return 0.0

        source_min, source_max = min(source_values), max(source_values)
        target_min, target_max = min(target_values), max(target_values)

        # Calculate intersection and union of ranges
        intersection_min = max(source_min, target_min)
        intersection_max = min(source_max, target_max)
        union_min = min(source_min, target_min)
        union_max = max(source_max, target_max)

        if intersection_max < intersection_min:
            # No overlap
            return 0.0

        intersection_size = intersection_max - intersection_min
        union_size = union_max - union_min

        if union_size == 0:
            return 1.0 if intersection_size == 0 else 0.0

        return intersection_size / union_size

    # ========================================================================
    # Algorithm 4: Distribution Shape Matching
    # ========================================================================

    def distribution_similarity(
        self, source_values: List[float], target_values: List[float]
    ) -> float:
        """Calculate distribution similarity using Kolmogorov-Smirnov test.

        Returns 1 - KS statistic (higher = more similar distributions).
        """
        if len(source_values) < 3 or len(target_values) < 3:
            # Need at least 3 samples for meaningful KS test
            return 0.0

        try:
            # Perform KS test
            ks_statistic, _ = stats.ks_2samp(source_values, target_values)
            # Convert to similarity score (1 - distance)
            return max(0.0, 1.0 - ks_statistic)
        except Exception:
            return 0.0

    # ========================================================================
    # Algorithm 5: Context Analysis
    # ========================================================================

    def context_similarity(self, source: str, target: str) -> Tuple[float, str]:
        """Analyze naming patterns and context clues.

        Heuristics:
        - Same prefix/suffix (e.g., 'train_acc' vs 'val_acc')
        - Abbreviation patterns (e.g., 'hp_lr' vs 'lr')
        - Common transformations (underscore vs camelCase)

        Returns:
            (score, reason): score between 0.0 and 1.0
        """
        source_lower = source.lower()
        target_lower = target.lower()

        reasons = []
        score = 0.0

        # Check for common prefixes/suffixes
        common_prefixes = ["train_", "val_", "test_", "hp_", "param_", "metric_"]
        common_suffixes = ["_loss", "_acc", "_rate", "_size", "_count"]

        source_base = source_lower
        target_base = target_lower

        for prefix in common_prefixes:
            if source_lower.startswith(prefix):
                source_base = source_lower[len(prefix) :]
            if target_lower.startswith(prefix):
                target_base = target_lower[len(prefix) :]

        for suffix in common_suffixes:
            if source_lower.endswith(suffix):
                source_base = source_lower[: -len(suffix)]
            if target_lower.endswith(suffix):
                target_base = target_lower[: -len(suffix)]

        if source_base == target_base:
            score = 0.9
            reasons.append(f"Same base name: {source_base}")

        # Check for abbreviation patterns
        if target_lower in source_lower or source_lower in target_lower:
            score = max(score, 0.7)
            reasons.append("One name contains the other")

        # Check for underscore vs camelCase
        source_parts = source_lower.replace("_", " ").split()
        target_parts = target_lower.replace("_", " ").split()
        if set(source_parts) == set(target_parts):
            score = max(score, 0.8)
            reasons.append("Same words, different formatting")

        reason = "; ".join(reasons) if reasons else "No context match"
        return score, reason

    # ========================================================================
    # Combined Scoring & Suggestion Generation
    # ========================================================================

    def calculate_combined_score(
        self,
        source: str,
        target: str,
        source_values: Optional[List[float]] = None,
        target_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Calculate weighted combined confidence score from all algorithms.

        Weights:
        - Fuzzy: 0.25
        - Semantic: 0.30
        - Value Range: 0.20
        - Distribution: 0.15
        - Context: 0.10

        Returns dict with scores and details.
        """
        # Algorithm scores
        fuzzy_score = self.fuzzy_similarity(source, target)
        semantic_score, semantic_group = self.semantic_similarity(source, target)
        context_score, context_reason = self.context_similarity(source, target)

        value_range_score = 0.0
        distribution_score = 0.0

        if source_values and target_values:
            # Convert to float if needed
            source_floats = [
                float(v) for v in source_values if isinstance(v, (int, float))
            ]
            target_floats = [
                float(v) for v in target_values if isinstance(v, (int, float))
            ]

            if source_floats and target_floats:
                value_range_score = self.value_range_similarity(
                    source_floats, target_floats
                )
                distribution_score = self.distribution_similarity(
                    source_floats, target_floats
                )

        # Weighted combination
        weights = {
            "fuzzy": 0.25,
            "semantic": 0.30,
            "value_range": 0.20,
            "distribution": 0.15,
            "context": 0.10,
        }

        combined_score = (
            fuzzy_score * weights["fuzzy"]
            + semantic_score * weights["semantic"]
            + value_range_score * weights["value_range"]
            + distribution_score * weights["distribution"]
            + context_score * weights["context"]
        )

        # Determine primary algorithm
        algorithm_scores = {
            "fuzzy": fuzzy_score,
            "semantic": semantic_score,
            "value_range": value_range_score,
            "distribution": distribution_score,
            "context": context_score,
        }
        primary_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])[0]

        # Generate human-readable reason
        reasons = []
        if semantic_score > 0.5:
            reasons.append(f"Semantic match (group: {semantic_group})")
        if fuzzy_score > 0.8:
            reasons.append(f"String similarity: {fuzzy_score:.2f}")
        if value_range_score > 0.6:
            reasons.append(f"Value range overlap: {value_range_score:.2f}")
        if context_score > 0.5:
            reasons.append(context_reason)

        reason = (
            "; ".join(reasons)
            if reasons
            else f"Low confidence match ({primary_algorithm})"
        )

        return {
            "combined_score": combined_score,
            "primary_algorithm": primary_algorithm,
            "reason": reason,
            "details": {
                "fuzzy": fuzzy_score,
                "semantic": semantic_score,
                "value_range": value_range_score,
                "distribution": distribution_score,
                "context": context_score,
                "semantic_group": semantic_group,
                "context_reason": context_reason,
            },
        }

    def generate_suggestions(
        self,
        project: str,
        source_columns: List[str],
        target_columns: List[str],
        source_data: Dict[str, List[Any]],
        target_data: Dict[str, List[Any]],
        column_type: str = "hyperparam",
        min_confidence: float = 0.5,
    ) -> List[ColumnSuggestion]:
        """Generate column mapping suggestions for unmapped columns.

        Args:
            project: Project name
            source_columns: New/unmapped column names
            target_columns: Existing unified column names
            source_data: Dict of source_column -> [values]
            target_data: Dict of target_column -> [values]
            column_type: 'hyperparam' or 'metric'
            min_confidence: Minimum confidence threshold

        Returns:
            List of ColumnSuggestion objects sorted by confidence (desc)
        """
        suggestions = []

        for source_col in source_columns:
            best_match = None
            best_score = 0.0

            for target_col in target_columns:
                # Skip if source and target are the same (don't suggest mapping to itself)
                if source_col.lower() == target_col.lower():
                    continue

                # Calculate scores
                result = self.calculate_combined_score(
                    source_col,
                    target_col,
                    source_data.get(source_col),
                    target_data.get(target_col),
                )

                if result["combined_score"] > best_score:
                    best_score = result["combined_score"]
                    best_match = (target_col, result)

            # Only suggest if above threshold
            if best_match and best_score >= min_confidence:
                target_col, result = best_match
                suggestions.append(
                    ColumnSuggestion(
                        source_column=source_col,
                        target_column=target_col,
                        confidence_score=result["combined_score"],
                        algorithm=result["primary_algorithm"],
                        reason=result["reason"],
                        column_type=column_type,
                        similarity_details=result["details"],
                    )
                )

        # Sort by confidence descending
        suggestions.sort(key=lambda x: x.confidence_score, reverse=True)
        return suggestions

    # ========================================================================
    # Database Operations
    # ========================================================================

    def get_mappings(
        self, project: str, column_type: Optional[str] = None, active_only: bool = True
    ) -> List[ColumnMapping]:
        """Get all column mappings for a project"""
        query = self.db.query(ColumnMapping).filter(ColumnMapping.project == project)

        if column_type:
            query = query.filter(ColumnMapping.column_type == column_type)

        if active_only:
            query = query.filter(ColumnMapping.is_active == True)

        return query.all()

    def create_mapping(
        self,
        project: str,
        source_column: str,
        target_column: str,
        column_type: str,
        mapping_method: str = "manual",
        confidence_score: Optional[float] = None,
        algorithm: Optional[str] = None,
        mapping_metadata: Optional[str] = None,
    ) -> ColumnMapping:
        """Create a new column mapping"""
        mapping = ColumnMapping(
            project=project,
            source_column=source_column,
            target_column=target_column,
            column_type=column_type,
            mapping_method=mapping_method,
            confidence_score=confidence_score,
            algorithm=algorithm,
            mapping_metadata=mapping_metadata,
        )
        self.db.add(mapping)
        self.db.commit()
        self.db.refresh(mapping)
        return mapping

    def update_mapping(
        self,
        mapping_id: int,
        target_column: Optional[str] = None,
        mapping_method: Optional[str] = None,
        is_active: Optional[bool] = None,
        mapping_metadata: Optional[str] = None,
    ) -> Optional[ColumnMapping]:
        """Update an existing column mapping"""
        mapping = (
            self.db.query(ColumnMapping).filter(ColumnMapping.id == mapping_id).first()
        )

        if not mapping:
            return None

        if target_column is not None:
            mapping.target_column = target_column
        if mapping_method is not None:
            mapping.mapping_method = mapping_method
        if is_active is not None:
            mapping.is_active = is_active
        if mapping_metadata is not None:
            mapping.mapping_metadata = mapping_metadata

        self.db.commit()
        self.db.refresh(mapping)
        return mapping

    def delete_mapping(self, mapping_id: int) -> bool:
        """Delete a column mapping"""
        mapping = (
            self.db.query(ColumnMapping).filter(ColumnMapping.id == mapping_id).first()
        )

        if not mapping:
            return False

        self.db.delete(mapping)
        self.db.commit()
        return True

    def apply_mappings(
        self, project: str, data: Dict[str, Any], column_type: str = "hyperparam"
    ) -> Dict[str, Any]:
        """Apply column mappings to transform data.

        Args:
            project: Project name
            data: Dict with original column names as keys
            column_type: 'hyperparam' or 'metric'

        Returns:
            Dict with unified column names
        """
        mappings = self.get_mappings(project, column_type, active_only=True)

        # Create mapping dict: source -> target
        mapping_dict = {m.source_column: m.target_column for m in mappings}

        # Apply mappings
        transformed = {}
        for key, value in data.items():
            new_key = mapping_dict.get(key, key)  # Use mapping or original
            transformed[new_key] = value

        return transformed
