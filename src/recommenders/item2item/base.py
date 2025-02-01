from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame, read_csv
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import LightFMWrapperModel

from src.constants import ItemsFeatureTopKConfig
from src.recommenders.feature_processors import FeaturePreparer

Columns.Datetime = 'last_watch_dt'


class BaseI2IRecommender(ABC):
    """Базовый класс для I2I рекомендательных систем."""

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
        k: int = 10,
        viewed_items: list[int] | None = None,
    ) -> DataFrame:
        recos: DataFrame = self.model.recommend_to_items(
            target_items=viewed_items,
            dataset=self.dataset,
            k=k,
            filter_itself=True,
        )

        merged_recos = recos.merge(
            self.items[[Columns.Item, 'title']], on=Columns.Item
        ).drop_duplicates(subset=Columns.Item)

        return merged_recos.sort_values(
            by=[Columns.Rank, Columns.Score], ascending=[True, False]
        )

    def _load_data(
        self,
        items_path: str,
        users_path: str,
        interactions_path: str,
    ) -> None:
        self.items = read_csv(items_path)
        self.users = read_csv(users_path)
        self.interactions = read_csv(interactions_path)

        # Установка весов
        self.interactions[Columns.Weight] = np.where(
            self.interactions['watched_pct'] > 20, 3, 1
        )
        self.original_user_ids = set(self.users[Columns.User].values)

    @abstractmethod
    def _load_model(self, model_path: str) -> None:
        """Абстрактный метод для загрузки конкретной модели."""
        pass

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
