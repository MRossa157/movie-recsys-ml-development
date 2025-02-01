from pydantic import BaseModel


class UserFeatures(BaseModel):
    age: str
    sex: str
    income: str
    kids_flg: bool = False


class Features(BaseModel):
    items: list[int]
    user_features: UserFeatures


egor_features = Features(
    # items=[14804, 7693, 11115, 8148, 16382, 4072, 898],
    items=[14804, 11115, 16382, 4072],
    user_features=UserFeatures(
        age='age_18_24',
        sex='М',
        income='income_60_90',
    ),
)
dmasta_features = Features(
    items=[5583, 8270, 9865, 9773, 12516, 13632, 7250],
    user_features=UserFeatures(
        age='age_18_24',
        sex='М',
        income='income_40_60',
    ),
)
katya_features = Features(
    items=[2134, 14177, 10994, 12057, 12842, 13720, 14320, 5533, 10085, 6870],
    user_features=UserFeatures(
        age='age_18_24',
        sex='Ж',
        income='income_0_20',
    )
)
