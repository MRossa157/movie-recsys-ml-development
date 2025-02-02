from __future__ import annotations

import os
from typing import TYPE_CHECKING

from rectools.models.implicit_als import ImplicitALSWrapperModelConfig
from rectools.models.serialization import load_model

from src.recommenders.item2item.base import BaseI2IRecommender

if TYPE_CHECKING:
    from rectools.models import ImplicitALSWrapperModel


class ALSRecommender(BaseI2IRecommender):
    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file {model_path} not found')

        self.model: ImplicitALSWrapperModel = load_model(model_path)

        if not isinstance(
            self.model.config_class, ImplicitALSWrapperModelConfig
        ):
            raise TypeError('Invalid model format')


if __name__ == '__main__':
    from src.mock_user_features import egor_features
    recommender = ALSRecommender(
        model_path=r'src\models\als\20250201_19-07-52',
        items_path=r'src\datasets\items_processed.csv',
        users_path=r'src\datasets\users_processed.csv',
        interactions_path=r'src\datasets\interactions_processed.csv',
    )

    recommendations = recommender.recommend(
        viewed_items=egor_features.items,
        k=3,
    )

    print(recommendations)
