from __future__ import annotations

from abc import abstractmethod
from typing import Any

from numpy import where
from pandas import DataFrame
from rectools import Columns
from rectools.dataset import Dataset, IdMap, Interactions

from src.constants import ItemsFeatureTopKConfig
from src.recommenders.feature_processors import FeaturePreparer

Columns.Datetime = 'last_watch_dt'


class BaseRecommender:
    """Базовый класс для U2I рекомендательных систем."""

    def __init__(
        self,
        model_path: str,
        items: DataFrame,
        users: DataFrame,
        interactions: DataFrame,
    ) -> None:
        self._load_model(model_path)

        self.items = items
        self.users = users

        _feature_preparer = FeaturePreparer({
            'director_top_k': ItemsFeatureTopKConfig.DIRECTORS_TOP_K,
            'studio_top_k': ItemsFeatureTopKConfig.STUDIOS_TOP_K,
        })

        # Установка весов
        interactions[Columns.Weight] = where(
            interactions['watched_pct'] > 20, 3, 1
        )

        self.interactions = interactions

        dataset = Dataset.construct(
            interactions_df=interactions,
            user_features_df=_feature_preparer.prepare_user_features(users),
            item_features_df=_feature_preparer.prepare_item_features(items),
            cat_user_features=_feature_preparer.get_user_feature_names(),
            cat_item_features=_feature_preparer.get_item_feature_names(),
        )

        self.item_id_map = dataset.item_id_map

    def recommend(
        self,
        viewed_items: list[int],
        k: int = 10,
        user_features: dict[str, Any] | None = None,
    ) -> DataFrame:
        interactions_df = DataFrame({"item_id": viewed_items})
        interactions_df[Columns.Weight] = 3
        interactions_df[Columns.Datetime] = "2022-02-02"
        interactions_df[Columns.User] = "user"
        user_id_map = IdMap.from_values(interactions_df[Columns.User])
        interactions = Interactions.from_raw(
            interactions_df,
            user_id_map,
            self.item_id_map,
        )
        dataset = Dataset(user_id_map, self.item_id_map, interactions)

        recos = self.model.recommend(
            users=user_id_map.external_ids,
            dataset=dataset,
            k=k,
            filter_viewed=True,
        )

        return recos

    @abstractmethod
    def _load_model(self, model_path: str) -> None:
        """Абстрактный метод для загрузки конкретной модели."""
        pass

    def add_titles(self, recos: DataFrame) -> DataFrame:
        return recos.merge(
            self.items[[Columns.Item, 'title']], on=Columns.Item, how='left'
        ).sort_values(Columns.Rank)
