from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
from rectools import Columns


class FeatureType(str, Enum):
    ITEM = 'item'
    USER = 'user'


class BaseFeatureProcessor(ABC):
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class GenreProcessor(BaseFeatureProcessor):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        data['genre'] = (
            data['genres'].str.replace(', ', ',', regex=False).str.split(',')
        )
        return data[[Columns.Item, 'genre']].explode('genre')


class DirectorProcessor(BaseFeatureProcessor):
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        data['directors'] = (
            data['directors'].str.replace(', ', ',', regex=False).str.split(',')
        )
        top_directors = (
            data['directors'].explode().value_counts().head(self.top_k).index
        )
        data['director'] = data['directors'].apply(
            lambda x: [d if d in top_directors else 'other_director' for d in x]
        )
        return data[[Columns.Item, 'director']].explode('director')


class StudioProcessor(BaseFeatureProcessor):
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        def replace_rare_studios(studio_list):
            return [
                studio if studio in top_studios else 'other_studio'
                for studio in studio_list
            ]

        data['studio'] = data['studios'].str.split(r',\s*')

        top_studios = (
            data['studio'].explode().value_counts().head(self.top_k).index
        )
        data['studio'] = data['studio'].apply(replace_rare_studios)
        return data[[Columns.Item, 'studio']].explode('studio')


class SimpleFeatureProcessor(BaseFeatureProcessor):
    def __init__(
        self,
        feature_name: str,
        feature_type: FeatureType,
    ) -> None:
        self.feature_name = feature_name
        if feature_type == FeatureType.ITEM:
            self.id_col = Columns.Item
        elif feature_type == FeatureType.USER:
            self.id_col = Columns.User

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[[self.id_col, self.feature_name]]


class FeaturePreparer:
    def __init__(self, config: dict[str, int]) -> None:
        self.item_processors: dict[str, BaseFeatureProcessor] = {
            'genre': GenreProcessor(),
            'director': DirectorProcessor(config['director_top_k']),
            'studios': StudioProcessor(config['studio_top_k']),
            'content_type': SimpleFeatureProcessor(
                'content_type',
                FeatureType.ITEM,
            ),
            'countries': SimpleFeatureProcessor(
                'countries',
                FeatureType.ITEM,
            ),
            'release_decade': SimpleFeatureProcessor(
                'release_decade',
                FeatureType.ITEM,
            ),
            'age_rating': SimpleFeatureProcessor(
                'age_rating',
                FeatureType.ITEM,
            ),
        }

        self.user_processors: dict[str, BaseFeatureProcessor] = {
            'sex': SimpleFeatureProcessor('sex', FeatureType.USER),
            'age': SimpleFeatureProcessor('age', FeatureType.USER),
            'income': SimpleFeatureProcessor('income', FeatureType.USER)
        }

    def get_item_feature_names(self) -> list[str]:
        return list(self.item_processors.keys())

    def get_user_feature_names(self) -> list[str]:
        return list(self.user_processors.keys())

    def prepare_item_features(self, items: pd.DataFrame) -> pd.DataFrame:
        return self._prepare_features(
            data=items,
            processors=self.item_processors,
        )

    def prepare_user_features(self, users: pd.DataFrame) -> pd.DataFrame:
        return self._prepare_features(
            data=users,
            processors=self.user_processors,
        )

    def _prepare_features(
        self,
        data: pd.DataFrame,
        processors: dict[str, BaseFeatureProcessor],
    ) -> pd.DataFrame:
        features = []
        for feature_name, processor in processors.items():
            feature_df = processor.process(data.copy())
            feature_df.columns = ['id', 'value']
            feature_df['feature'] = feature_name
            features.append(feature_df)
        return pd.concat(features)
