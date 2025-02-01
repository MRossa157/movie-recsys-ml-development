from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

from src.recommenders.base import BaseRecommender

if TYPE_CHECKING:
    from rectools.models import LightFMWrapperModel

Columns.Datetime = 'last_watch_dt'


class LightFMRecommender(BaseRecommender):
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        viewed_items: list[int] | None = None,
        user_features: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if user_id not in self.original_user_ids:
            if not user_features:
                raise ValueError('User features required for new users')
            if not viewed_items:
                raise ValueError('Viewed items features required for new users')

            return self._recommend_for_new_user(
                user_id,
                k,
                viewed_items,
                user_features,
            )

        recos = self.model.recommend(
            users=[user_id], dataset=self.dataset, k=k, filter_viewed=True
        )

        return self._add_titles(recos)

    def _recommend_for_new_user(
        self,
        user_id: int,
        k: int,
        viewed_items: list[int],
        user_features: dict[str, Any],
    ) -> pd.DataFrame:
        temp_users, temp_interactions = self._create_temp_data(
            user_id,
            user_features,
            viewed_items,
        )

        temp_dataset = self._build_temp_dataset(temp_users, temp_interactions)

        recos = self.model.recommend(
            users=[user_id], dataset=temp_dataset, k=k, filter_viewed=True
        )

        return self._add_titles(recos)

    def _create_temp_data(
        self,
        user_id: int,
        user_features: dict[str, Any],
        viewed_items: list[int],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        new_user = pd.DataFrame([
            {
                **user_features,
                Columns.User: user_id,
            }
        ])

        new_interactions = pd.DataFrame({
            Columns.User: user_id,
            Columns.Item: viewed_items,
            Columns.Datetime: pd.Timestamp.now().date(),
            'total_dur': np.nan,
            'watched_pct': 100.0,
            Columns.Weight: 3,
        })

        temp_users = pd.concat([self.users, new_user], ignore_index=True)
        temp_interactions = pd.concat(
            [self.interactions, new_interactions],
            ignore_index=True,
        )

        return temp_users, temp_interactions

    def _build_temp_dataset(
        self,
        temp_users: pd.DataFrame,
        temp_interactions: pd.DataFrame,
    ) -> Dataset:
        # Готовим фичи для временного датасета
        user_features = self.feature_preparer.prepare_user_features(temp_users)
        item_features = self.feature_preparer.prepare_item_features(self.items)

        return Dataset.construct(
            interactions_df=temp_interactions,
            user_features_df=user_features,
            item_features_df=item_features,
            cat_user_features=self.cat_user_features,
            cat_item_features=self.cat_item_features,
        )

    def _add_titles(self, recos: pd.DataFrame) -> pd.DataFrame:
        return recos.merge(
            self.items[[Columns.Item, 'title']], on=Columns.Item, how='left'
        ).sort_values(Columns.Rank)

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file {model_path} not found')

        with open(model_path, 'rb') as f:
            self.model: LightFMWrapperModel = pickle.load(f)


if __name__ == '__main__':
    from src.mock_user_features import egor_features as mock_user

    recommender = LightFMRecommender(
        model_path=r'src\models\lightfm\20250128_15-52-34',
        items_path=r'src\datasets\items_processed.csv',
        users_path=r'src\datasets\users_processed.csv',
        interactions_path=r'src\datasets\interactions_processed.csv',
    )

    recommendations = recommender.recommend(
        user_id=1100000,
        viewed_items=mock_user.items,
        user_features=mock_user.user_features.dict(),
        k=10,
    )

    print(recommendations)
