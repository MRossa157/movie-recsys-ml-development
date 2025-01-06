from abc import ABC, abstractmethod

# TODO дописать тип возвращения функции


class BaseRecommender(ABC):

    @abstractmethod
    def __init__(
        self,
        model_path: str,
    ) -> None: ...

    @abstractmethod
    def get_recommendation(
        self,
    ) -> any:
        raise NotImplementedError

    @abstractmethod
    def get_recommendation_for_new_user(self) -> any:
        raise NotImplementedError
