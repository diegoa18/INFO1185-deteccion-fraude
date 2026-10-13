"""Selección de características compatible con pipelines de scikit-learn."""

from .feature_selection import make_mutual_information_selector, selected_feature_names

__all__ = ["make_mutual_information_selector", "selected_feature_names"]
