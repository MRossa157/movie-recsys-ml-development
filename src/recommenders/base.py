from abc import abstractmethod

from numpy import where
from pandas import DataFrame, read_csv
from rectools import Columns
from rectools.dataset import Dataset

from src.constants import ItemsFeatureTopKConfig
from src.recommenders.feature_processors import FeaturePreparer


class BaseRecommender:
    """Базовый класс для U2I рекомендательных систем."""

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
        self, user_id: int, viewed_items: list[int], k: int
    ) -> DataFrame:
        raise NotImplementedError

    def _load_data(
        self,
        items_path: str,
        users_path: str,
        interactions_path: str,
    ) -> None:
        self.items: DataFrame = read_csv(items_path)
        self.users: DataFrame = read_csv(users_path)
        self.interactions: DataFrame = read_csv(interactions_path)
        # Установка весов
        self.interactions[Columns.Weight] = where(
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
