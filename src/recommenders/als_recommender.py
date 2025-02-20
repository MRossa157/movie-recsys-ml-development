from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pandas as pd
from implicit.cpu.als import (
    AlternatingLeastSquares as CPUAlternatingLeastSquares,
)
from implicit.gpu.als import (
    AlternatingLeastSquares as GPUAlternatingLeastSquares,
)
from rectools import Columns
from rectools.dataset import Dataset, IdMap, Interactions
from rectools.models.serialization import load_model

from src.recommenders.base import BaseRecommender

if TYPE_CHECKING:
    from rectools.models import ImplicitALSWrapperModel


class ALSRecommender(BaseRecommender):
    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file {model_path} not found')

        self.model: ImplicitALSWrapperModel = load_model(model_path)

        if not isinstance(
            self.model.model,
            (GPUAlternatingLeastSquares, CPUAlternatingLeastSquares),
        ):
            raise TypeError('Invalid model format')


if __name__ == '__main__':
    from pandas import read_csv

    from src.mock_user_features import egor_features as mock_user

    items = read_csv(r'src\datasets\items_processed.csv')
    users = read_csv(r'src\datasets\users_processed.csv')
    interactions = read_csv(r'src\datasets\interactions_processed.csv')

    recommender = ALSRecommender(
        model_path=r'src\models\als\20250220_16-33-19',
        items=items,
        users=users,
        interactions=interactions,
    )

    recommendations = recommender.recommend(
        viewed_items=mock_user.items,
        k=10,
        user_features=mock_user.user_features.dict(),
    )

    print(recommender.add_titles(recommendations))
