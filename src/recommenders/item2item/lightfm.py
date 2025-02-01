from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

from rectools import Columns

from recommenders.item2item.base import BaseI2IRecommender

if TYPE_CHECKING:
    from rectools.models import LightFMWrapperModel

Columns.Datetime = 'last_watch_dt'


class LightFMRecommender(BaseI2IRecommender):
    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file {model_path} not found')

        with open(model_path, 'rb') as f:
            self.model: LightFMWrapperModel = pickle.load(f)


if __name__ == '__main__':
    recommender = LightFMRecommender(
        model_path=r'src\models\als\20250125_18-22-26',
        items_path=r'src\datasets\items_processed.csv',
        users_path=r'src\datasets\users_processed.csv',
        interactions_path=r'src\datasets\interactions_processed.csv',
    )

    recommendations = recommender.recommend(
        user_id=1100000,
        # viewed_items=[14804, 7693, 11115, 8148, 16382, 4072, 898],  # Егор
        viewed_items=[5583, 8270, 9865, 9773, 12516, 13632, 7250, ],  # Димаста
        user_features={
            'age': 'age_18_24',
            'sex': 'М',
            'income': 'income_0_20',
            'kids_flg': False,
        },
        k=10,
    )

    print(recommendations)
