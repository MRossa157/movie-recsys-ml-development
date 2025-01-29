from pandas import DataFrame


class BaseRecommender:
    def recommend(
        self, user_id: int, viewed_items: list[int], k: int
    ) -> DataFrame:
        raise NotImplementedError
