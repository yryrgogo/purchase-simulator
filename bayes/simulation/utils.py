from typing import List

import pandas as pd
import pendulum


def group_by_weekday(today: pendulum.DateTime, df: pd.DataFrame) -> List[List[int]]:  # type: ignore
    """直近の日曜日から過去28日分のデータを曜日ごとにグループ化する
    pendulum の weekday は以下のようになっている
    0	Monday
    1	Tuesday
    2	Wednesday
    3	Thursday
    4	Friday
    5	Saturday
    6	Sunday

    Parameters
    -------
    today : pendulum.DateTime
        今日の日付
    df : pd.DataFrame
        今日から過去28日分のデータ

    Returns
    -------
    List[List[int]]
        今日から過去28日分のデータを曜日ごとにグループ化したデータ
        配列のインデックスは曜日を表す
    """
    end_date = today.previous(pendulum.SUNDAY)  # type: ignore
    start_date = end_date.subtract(days=27)

    weekday_purchase_data: List[List[int]] = [[] for _ in range(7)]

    for i in range((end_date - start_date).in_days() + 1):  # type: ignore
        date = start_date.add(days=i)
        weekday = date.weekday()

        # Get the count for the current date
        count: List[int] = df.loc[df["date"] == date.to_date_string(), "count"].values  # type: ignore

        if len(count) > 0:
            weekday_purchase_data[weekday].append(count[0])
        else:
            weekday_purchase_data[weekday].append(0)

    return weekday_purchase_data


def get_weekday_name(weekday: int) -> str:
    """曜日の番号を曜日の名前に変換する

    Parameters
    -------
    weekday : int
        曜日の番号

    Returns
    -------
    str
        曜日の名前
    """
    weekday_names = ["月", "火", "水", "木", "金", "土", "日"]

    return weekday_names[weekday]
