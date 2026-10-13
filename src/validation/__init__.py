"""Configuración reutilizable de validación cruzada."""

from .cross_validation import make_stratified_kfold, materialize_stratified_folds

__all__ = ["make_stratified_kfold", "materialize_stratified_folds"]
