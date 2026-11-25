"""Golden dataset: human-labeled ground truth for the classifier.

Public API for the eval runner:
    from app.golden import load_golden, GoldenDataset, GoldenCase
"""

from app.golden.io import GoldenDatasetError, GoldenIssue, dump_golden, find_issues, load_golden, save_golden
from app.golden.models import Difficulty, GoldenCase, GoldenDataset
from app.golden.stats import GoldenStats, compute_stats

__all__ = [
    "Difficulty",
    "GoldenCase",
    "GoldenDataset",
    "GoldenDatasetError",
    "GoldenIssue",
    "GoldenStats",
    "compute_stats",
    "dump_golden",
    "find_issues",
    "load_golden",
    "save_golden",
]
