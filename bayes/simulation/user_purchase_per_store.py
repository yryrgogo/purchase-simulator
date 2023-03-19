from typing import List

import numpy as np
import pandas as pd
import pendulum
import torch

from models.user_store_purchase import BayesianModel, PosteriorResult
from simulation.utils import group_by_weekday


class UserPurchasePerStoreSimulator:
    """
    ユーザーの店舗での購買回数をシミュレートする。
    現状は曜日別の購買回数しか出力しない。
    """

    def __init__(self, user_id: str):
        """
        Parameters
        -------
        user_id : str
            ユーザーID
        """
        self.user_id = user_id
        self.posterior_per_weekdays: List[PosteriorResult] = []
        self.trials = 1000
        self.simulated_weekday_purchase_counts: List[float] = []

    def _fetch_daily_data(self, num_weeks: int) -> List[torch.Tensor]:
        """ユーザーの曜日別の購買回数を取得する

        Parameters
        -------
        num_weeks : int
            直近何週間のデータを取得するか

        Returns
        -------
        List[torch.Tensor]
            ユーザーの曜日別の購買回数の配列
        """
        # user_id に対応するデータを取得する
        # 直近4週間の各店舗の購買頻度を取得する。当週は除外して、直近の日曜日から数えて4週間分のデータを取得する
        # 曜日別に配列に値を格納する

        df = pd.DataFrame(
            {
                "date": [
                    "2023-02-19",
                    "2023-02-20",
                    "2023-02-21",
                    "2023-02-22",
                    "2023-02-23",
                    "2023-02-24",
                    "2023-02-25",
                    "2023-02-26",
                    "2023-02-27",
                    "2023-02-28",
                    "2023-03-01",
                    "2023-03-02",
                    "2023-03-03",
                    "2023-03-04",
                    "2023-03-05",
                    "2023-03-06",
                    "2023-03-07",
                    "2023-03-08",
                    "2023-03-09",
                    "2023-03-10",
                    "2023-03-11",
                    "2023-03-12",
                    "2023-03-13",
                    "2023-03-14",
                    "2023-03-15",
                    "2023-03-16",
                    "2023-03-17",
                    "2023-03-18",
                ],
                "count": [
                    5,
                    3,
                    2,
                    6,
                    5,
                    3,
                    7,
                    3,
                    8,
                    1,
                    4,
                    6,
                    3,
                    9,
                    2,
                    4,
                    5,
                    6,
                    7,
                    2,
                    3,
                    5,
                    8,
                    9,
                    3,
                    5,
                    6,
                    7,
                ],
            }
        )

        grouped_by_weekday_data = group_by_weekday(pendulum.now(), df)
        grouped_by_weekday_tensor: List[torch.Tensor] = []
        for weekday_data in grouped_by_weekday_data:
            grouped_by_weekday_tensor.append(torch.tensor(weekday_data))
        return grouped_by_weekday_tensor

    def simulate(self):
        """ """
        num_weeks = 4
        grouped_by_weekday_tensor = self._fetch_daily_data(num_weeks)

        print(f"ユーザー ID: {self.user_id}, 週数: {num_weeks}")

        # 各曜日のデータをモデルに渡して、事後分布を取得する
        for tensor in grouped_by_weekday_tensor:
            model = BayesianModel(tensor, self.trials)
            posterior_samples = model.run()
            self.posterior_per_weekdays.append(PosteriorResult(posterior_samples))

        result: List[float] = []
        for posterior in self.posterior_per_weekdays:
            result.append(np.round(posterior.median, 2))  # type: ignore

        self.simulated_weekday_purchase_counts = result
