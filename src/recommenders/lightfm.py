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
    from rectools.models import LightFMWrapperModel

Columns.Datetime = 'last_watch_dt'


class LightFMRecommender(BaseRecommender):
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
        self._init_base_dataset()

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        viewed_items: list[int] | None = None,
        user_features: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if user_id not in self.original_user_ids:
            return self._recommend_for_new_user(
                user_id,
                k,
                viewed_items,
                user_features,
            )

        recos = self.model.recommend(
            users=[user_id],
            dataset=self.dataset,
            k=k,
            filter_viewed=True
        )

        return self._add_titles(recos)

    def _recommend_for_new_user(
        self,
        user_id: int,
        k: int,
        viewed_items: list[int],
        user_features: dict[str, Any],
    ) -> pd.DataFrame:
        # Валидация входных данных
        if not user_features:
            raise ValueError('User features required for new users')
        if not viewed_items:
            raise ValueError('Viewed items features required for new users')

        # Создаем временные данные
        temp_users, temp_interactions = self._create_temp_data(
            user_id,
            user_features,
            viewed_items,
        )

        # Собираем временный датасет
        temp_dataset = self._build_temp_dataset(temp_users, temp_interactions)

        # Получаем рекомендации
        recos = self.model.recommend(
            users=[user_id],
            dataset=temp_dataset,
            k=k,
            filter_viewed=True
        )

        return self._add_titles(recos)

    def _create_temp_data(
        self,
        user_id: int,
        user_features: dict[str, Any],
        viewed_items: list[int],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Создаем временного пользователя
        new_user = pd.DataFrame([{
            **user_features,
            Columns.User: user_id,
        }])

        # Создаем временные взаимодействия
        new_interactions = pd.DataFrame({
            Columns.User: user_id,
            Columns.Item: viewed_items,
            Columns.Datetime: pd.Timestamp.now().date(),
            'total_dur': np.nan,
            'watched_pct': 100.0,
            Columns.Weight: 3,
        })

        # Объединяем с исходными данными
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
            self.items[[Columns.Item, 'title']],
            on=Columns.Item,
            how='left'
        ).sort_values(Columns.Rank)

    def _load_data(
        self,
        items_path: str,
        users_path: str,
        interactions_path: str,
    ) -> None:
        self.items = pd.read_csv(items_path)
        self.users = pd.read_csv(users_path)
        self.interactions = pd.read_csv(interactions_path)
        # Установка весов
        self.interactions[Columns.Weight] = np.where(
            self.interactions['watched_pct'] > 20, 3, 1
        )
        self.original_user_ids = set(self.users[Columns.User].values)

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file {model_path} not found')

        with open(model_path, 'rb') as f:
            self.model: LightFMWrapperModel = pickle.load(f)

    def _init_base_dataset(self) -> None:
        user_features = self.feature_preparer.prepare_user_features(self.users)
        item_features = self.feature_preparer.prepare_item_features(self.items)

        self.dataset = Dataset.construct(
            interactions_df=self.interactions,
            user_features_df=user_features,
            item_features_df=item_features,
            cat_user_features=self.cat_user_features,
            cat_item_features=self.cat_item_features,
        )


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
