from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

if TYPE_CHECKING:
    from rectools.models import ImplicitALSWrapperModel

Columns.Datetime = 'last_watch_dt'


class BaseRecommender:
    def recommend(
        self, user_id: int, viewed_items: list[int], k: int
    ) -> pd.DataFrame:
        raise NotImplementedError


class ALSRecommender(BaseRecommender):
    def __init__(
        self,
        model_path: str,
        items_path: str,
        users_path: str,
        interactions_path: str,
        cat_item_features: list[str],
        cat_user_features: list[str],
    ) -> None:
        self.cat_item_features = cat_item_features
        self.cat_user_features = cat_user_features

        self._load_data(items_path, users_path, interactions_path)
        self._load_model(model_path)
        self._init_dataset(self.cat_item_features, self.cat_user_features)

    def recommend(
        self,
        user_id: int,
        viewed_items: list[int],
        k: int = 10,
        user_features: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if user_id not in self.original_user_ids:
            if not user_features:
                raise ValueError('User features required for new users')
            self._add_new_user(user_id, user_features, viewed_items)

            self._init_dataset(self.cat_item_features, self.cat_user_features)

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
            self.items[['item_id', 'title']],
            on='item_id'
        ).sort_values('rank', ascending=False)

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
        self.original_user_ids = set(self.users['user_id'].values)
        self.original_item_ids = set(self.items['item_id'].values)

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file {model_path} not found')

        with open(model_path, 'rb') as f:
            self.model: ImplicitALSWrapperModel = pickle.load(f)

        if not hasattr(self.model, 'recommend'):
            raise ValueError('Invalid model format')

    def _init_dataset(
        self,
        cat_item_features: list[str],
        cat_user_features: list[str],
    ) -> None:
        self.dataset = Dataset.construct(
            interactions_df=self.interactions,
            user_features_df=self._prepare_features('user', cat_user_features),
            item_features_df=self._prepare_features('item', cat_item_features),
            cat_user_features=cat_user_features,
            cat_item_features=cat_item_features,
        )

    def _prepare_features(
        self, entity: str, features: list[str]
    ) -> pd.DataFrame:
        if entity == 'user':
            data = self.users
            id_col = Columns.User
        else:
            data = self.items
            id_col = Columns.Item

        feature_frames = []
        for feature in features:
            frame = data[[id_col, feature]].copy()
            frame.columns = ['id', 'value']
            frame['feature'] = feature
            feature_frames.append(frame)
        return pd.concat(feature_frames)

    def _add_new_user(
        self,
        user_id: int,
        user_features: dict[str, Any],
        viewed_items: list[int],
    ) -> None:
        new_user = pd.DataFrame([{**user_features, 'user_id': user_id}])
        self.users = pd.concat([self.users, new_user], ignore_index=True)

        new_interactions = pd.DataFrame({
            'user_id': user_id,
            'item_id': viewed_items,
            'last_watch_dt': pd.Timestamp.now().date(),
            'total_dur': np.nan,
            'watched_pct': 100.0,
            'weight': 3,
        })
        self.interactions = pd.concat([self.interactions, new_interactions])


if __name__ == '__main__':
    # Инициализация
    recommender = ALSRecommender(
        model_path=r'src\models\als\20250125_18-22-26',
        items_path=r'src\sandbox\items_processed.csv',
        users_path=r'src\sandbox\users_processed.csv',
        interactions_path=r'src\sandbox\interactions_processed.csv',
        cat_item_features=[
            'genre',
            'content_type',
            'countries',
            'release_decade',
            'age_rating',
            'studio',
            'director',
        ],
        cat_user_features=['sex', 'age', 'income'],
    )

    # Получение рекомендаций для нового пользователя
    recommendations = recommender.recommend(
        user_id=1100000,
        viewed_items=[14804, 7693, 11115, 8148, 16382, 4072, 898],
        user_features={
            'age': 'age_25_34',
            'sex': 'М',
            'income': 'income_60_90',
            'kids_flg': False
        },
        k=10
    )

    print(recommendations[['item_id', 'title', 'score']])
