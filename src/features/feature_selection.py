"""Selección de variables mediante información mutua."""

from __future__ import annotations

from functools import partial
from typing import Literal, Sequence

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from src.config import RANDOM_STATE


def make_mutual_information_selector(
    k: int | Literal["all"] = 10,
    *,
    random_state: int = RANDOM_STATE,
) -> SelectKBest:
    """Crea un selector ``SelectKBest`` basado en información mutua.

    Debe incorporarse a un pipeline y ajustarse en cada partición de entrenamiento de
    la validación cruzada. La información mutua identifica dependencias (también no
    lineales), pero no garantiza causalidad ni reemplaza la validación del modelo.
    """
    if k != "all" and (not isinstance(k, int) or k <= 0):
        raise ValueError("k debe ser un entero positivo o 'all'.")
    score_function = partial(mutual_info_classif, random_state=random_state)
    return SelectKBest(score_func=score_function, k=k)


def selected_feature_names(
    selector: SelectKBest,
    feature_names: Sequence[str] | None = None,
) -> list[str]:
    """Obtiene los nombres seleccionados una vez ajustado el selector.

    Si el selector se ajustó con un DataFrame, no hace falta entregar
    ``feature_names``. En otro caso se deben proporcionar en el mismo orden de entrada.
    """
    support = selector.get_support()
    if feature_names is None:
        if not hasattr(selector, "feature_names_in_"):
            raise ValueError(
                "Entregue feature_names si el selector no fue ajustado con un DataFrame."
            )
        feature_names = selector.feature_names_in_
    if len(feature_names) != len(support):
        raise ValueError("feature_names debe tener la misma longitud que las características.")
    return list(np.asarray(feature_names)[support])
