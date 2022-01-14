from datetime import datetime, time
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_recipes():
    return pd.read_csv("data/lunch_recipes.csv", parse_dates=["date"])


def process_recipes(recipes, large_dish_indicators=["pan", "rasp", "kom"], ignored_words=["servings", "recipe", "url", "dish"]):
    for indicator in large_dish_indicators:
        recipes[indicator] = recipes.recipe.apply(
            lambda text: tokenize(text).count(indicator) > 0)

    recipes = recipes.drop(columns=ignored_words)

    return recipes


def tokenize(recipe: str) -> List[str]:
    return [remove_punctuation(token).lower() for token in recipe.split()]


def remove_punctuation(token: str) -> str:
    return ''.join(character for character in token if character.isalnum())


def load_attendances():
    df = pd.read_csv("data/key_tag_logs.csv")
    df['timestamp2'] = df.timestamp.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['date'] = df.timestamp.apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d'))
    df['time'] = df.timestamp2.apply(lambda x: x.time())
    df['timestamp'] = df['timestamp2']
    df = df.drop('timestamp2', axis=1)

    result = pd.DataFrame(np.array(df.date), columns=['date']).drop_duplicates()

    for name in df.name.unique():
        lunchdates = []
        for datum in df.date.unique():
            df2 = df[df.name == name]
            df2 = df2[df2.date == datum]

            dataframe_check_in = df2[df2.event == "check in"]
            dataframe_check_in = dataframe_check_in[dataframe_check_in.time < time(12, 0, 0)]

            df_check_out = df2[df2.event == "check out"]
            df_check_out = df_check_out[df_check_out.time > time(12, 0, 0)]
            if df_check_out.shape[0] > 0 and dataframe_check_in.shape[0] > 0:
                lunchdates.append(datum)

        result[name] = result.date.apply(lambda x: 1 if x in list(lunchdates) else 0)

    return result


def load_dishwasher_log():
    return pd.read_csv("data/dishwasher_log.csv", parse_dates=['date'])


def train_model():
    recipes = (load_recipes()
               .pipe(process_recipes))
    attendance = load_attendances()
    dishwasher_log = load_dishwasher_log()

    df = recipes.merge(attendance,
                       on="date",
                       how="outer").merge(dishwasher_log).fillna(0)
    reg = LinearRegression(fit_intercept=False, positive=True) \
        .fit(df.drop(["dishwashers", "date"], axis=1),
             df["dishwashers"])
    return dict(zip(reg.feature_names_in_,
                    [round(c, 3) for c in reg.coef_]))


def run():
    print(train_model())


if __name__ == "__main__":
    run()
