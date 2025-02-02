from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from implicit.cpu.als import (
    AlternatingLeastSquares as CPUAlternatingLeastSquares,
)
from implicit.gpu.als import (
    AlternatingLeastSquares as GPUAlternatingLeastSquares,
)
from rectools import Columns
from rectools.models.serialization import load_model

from src.recommenders.base import BaseRecommender

if TYPE_CHECKING:
    from rectools.models import ImplicitALSWrapperModel


class ALSRecommender(BaseRecommender):
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
            users=[user_id], dataset=self.dataset, k=k, filter_viewed=True
        )

        return recos.merge(
            self.items[[Columns.Item, 'title']], on=Columns.Item
        ).sort_values(Columns.Rank)

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file {model_path} not found')

        self.model: ImplicitALSWrapperModel = load_model(model_path)

        if not isinstance(
            self.model.model,
            (GPUAlternatingLeastSquares, CPUAlternatingLeastSquares),
        ):
            raise TypeError('Invalid model format')

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
    from mock_user_features import egor_features as mock_user

    recommender = ALSRecommender(
        model_path=r'src\models\als\20250125_18-22-26',
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
