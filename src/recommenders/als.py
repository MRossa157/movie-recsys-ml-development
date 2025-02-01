from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

from src.constants import ItemsFeatureTopKConfig
from src.recommenders.base import BaseRecommender
from src.recommenders.feature_processors import FeaturePreparer

if TYPE_CHECKING:
    from rectools.models import ImplicitALSWrapperModel

Columns.Datetime = 'last_watch_dt'


class ALSRecommender(BaseRecommender):
    def __init__(
        self,
        model_path: str,
        items_path: str,
        users_path: str,
        interactions_path: str,
    ) -> None:
        self.feature_config = {
            'director_top_k': ItemsFeatureTopKConfig.DIRECTORS_TOP_K,
            'studio_top_k': ItemsFeatureTopKConfig.STUDIOS_TOP_K
        }

        self.feature_preparer = FeaturePreparer(self.feature_config)

        self.cat_item_features = self.feature_preparer.get_item_feature_names()
        self.cat_user_features = self.feature_preparer.get_user_feature_names()

        self._load_data(items_path, users_path, interactions_path)
        self._load_model(model_path)
        self._init_dataset()

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

            self._add_new_user(user_id, user_features, viewed_items)

            self._init_dataset()

            # Для нового пользователя требуется дообучение
            self.model.fit(self.dataset)

        # Получаем рекомендации
        recos = self.model.recommend(
            users=[user_id],
            dataset=self.dataset,
            k=k,
            filter_viewed=True
        )

        return recos.merge(
            self.items[[Columns.Item, 'title']],
            on=Columns.Item
        ).sort_values(Columns.Rank)

    def _load_data(
        self,
        items_path: str,
        users_path: str,
        interactions_path: str,
    ) -> None:
        self.items: pd.DataFrame = pd.read_csv(items_path)
        self.users: pd.DataFrame = pd.read_csv(users_path)
        self.interactions: pd.DataFrame = pd.read_csv(interactions_path)
        # Установка весов
        self.interactions[Columns.Weight] = np.where(
            self.interactions['watched_pct'] > 20, 3, 1
        )
        self.original_user_ids = set(self.users[Columns.User].values)

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file {model_path} not found')

        with open(model_path, 'rb') as f:
            self.model: ImplicitALSWrapperModel = pickle.load(f)

        if not hasattr(self.model, 'recommend'):
            raise ValueError('Invalid model format')

    def _init_dataset(self) -> None:
        item_features = self.feature_preparer.prepare_item_features(self.items)
        user_features = self.feature_preparer.prepare_user_features(self.users)

        self.dataset = Dataset.construct(
            interactions_df=self.interactions,
            user_features_df=user_features,
            item_features_df=item_features,
            cat_user_features=self.cat_user_features,
            cat_item_features=self.cat_item_features,
        )

    def _add_new_user(
        self,
        user_id: int,
        user_features: dict[str, Any],
        viewed_items: list[int],
    ) -> None:
        new_user = pd.DataFrame([{**user_features, Columns.User: user_id}])
        self.users = pd.concat([self.users, new_user], ignore_index=True)

        new_interactions = pd.DataFrame({
            Columns.User: user_id,
            Columns.Item: viewed_items,
            Columns.Datetime: pd.Timestamp.now().date(),
            'total_dur': np.nan,
            'watched_pct': 100.0,
            Columns.Weight: 3,
        })
        self.interactions = pd.concat([self.interactions, new_interactions])


if __name__ == '__main__':
    from mock_user_features import egor_features
    recommender = ALSRecommender(
        model_path=r'src\models\als\20250125_18-22-26',
        items_path=r'src\datasets\items_processed.csv',
        users_path=r'src\datasets\users_processed.csv',
        interactions_path=r'src\datasets\interactions_processed.csv',
    )

    recommendations = recommender.recommend(
        user_id=1100000,
        viewed_items=egor_features.items,
        user_features=egor_features.user_features.dict(),
    )

    print(recommendations[['item_id', 'title', 'score']])
