from __future__ import annotations

import os
from typing import TYPE_CHECKING

from rectools.models.lightfm import LightFM
from rectools.models.serialization import load_model

from src.recommenders.base import BaseRecommender

if TYPE_CHECKING:
    from rectools.models import LightFMWrapperModel


class LightFMRecommender(BaseRecommender):
    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file {model_path} not found')

        self.model: LightFMWrapperModel = load_model(model_path)

        if not isinstance(self.model.model, LightFM):
            raise TypeError('Invalid model format')


if __name__ == '__main__':
    from pandas import read_csv

    from src.mock_user_features import egor_features as mock_user

    items = read_csv(r'src\datasets\items_processed.csv')
    users = read_csv(r'src\datasets\users_processed.csv')
    interactions = read_csv(r'src\datasets\interactions_processed.csv')

    recommender = LightFMRecommender(
        model_path=r'src\models\lightfm\20250201_13-33-26_top_map10',
        items=items,
        users=users,
        interactions=interactions,
    )

    recommendations = recommender.recommend(
        viewed_items=mock_user.items,
        user_features=mock_user.user_features.dict(),
        k=10,
    )

    print(recommender.add_titles(recommendations))
