from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

from rectools import Columns

from src.recommenders.item2item.base import BaseI2IRecommender

if TYPE_CHECKING:
    from rectools.models import ImplicitALSWrapperModel

Columns.Datetime = 'last_watch_dt'


class ALSRecommender(BaseI2IRecommender):
    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file {model_path} not found')

        with open(model_path, 'rb') as f:
            self.model: ImplicitALSWrapperModel = pickle.load(f)

        if not hasattr(self.model, 'recommend'):
            raise ValueError('Invalid model format')


if __name__ == '__main__':
    from src.mock_user_features import egor_features
    recommender = ALSRecommender(
        model_path=r'src\models\als\20250125_18-22-26',
        items_path=r'src\datasets\items_processed.csv',
        users_path=r'src\datasets\users_processed.csv',
        interactions_path=r'src\datasets\interactions_processed.csv',
    )

    recommendations = recommender.recommend(
        viewed_items=egor_features.items,
        k=3,
    )

    print(recommendations)
