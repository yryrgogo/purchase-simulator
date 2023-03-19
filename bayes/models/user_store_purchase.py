from typing import Any, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS


class BayesianModel:
    def __init__(self, observed_data: torch.Tensor, num_trials: int):
        self.observed_data = observed_data
        self.num_trials = num_trials or 1000
        return

    def run(self) -> List[float]:
        posterior_samples = self._compute_posterior()
        return posterior_samples

    def _model(self, observed_data: torch.Tensor):
        k, theta = GammaDist.calculate_gamma_params(observed_data)
        lambda_latent = pyro.sample("lambda_latent", dist.Gamma(k, theta))  # type: ignore
        with pyro.plate("data", len(self.observed_data)):
            pyro.sample("obs", dist.Poisson(lambda_latent), obs=self.observed_data)  # type: ignore

    def _compute_posterior(self) -> List[float]:
        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(nuts_kernel, num_samples=self.num_trials, warmup_steps=200)

        mcmc.run(self.observed_data)  # type: ignore
        result = mcmc.get_samples()  # type: ignore
        if result is None:
            raise Exception("result is None")

        posterior_samples: List[float] = result["lambda_latent"].numpy()  # type: ignore

        return posterior_samples


class PosteriorResult:
    def __init__(self, posterior_samples: List[float]):
        self.posterior_samples = posterior_samples

        self.lower_credible_interval = np.percentile(  # type: ignore
            self.posterior_samples, 2.5
        )
        self.upper_credible_interval = np.percentile(  # type: ignore
            self.posterior_samples, 97.5
        )

    @property
    def mean(self) -> np.floating[Any]:
        return np.mean(self.posterior_samples)

    @property
    def median(self) -> np.floating[Any]:
        return np.median(self.posterior_samples)

    def show_stats(self):
        print("Mean:", self.mean)
        print("Median:", self.median)
        print(
            f"95% Credible Interval: [{self.lower_credible_interval}, {self.upper_credible_interval}]"
        )

    def create_scatter_plot(self) -> Any:
        # 小数点第2位までの確率に変換
        rounded_probabilities = np.round(self.posterior_samples, 2)  # type: ignore
        # 確率がサンプリングされた回数を計算
        unique_probabilities, sample_counts = np.unique(
            rounded_probabilities, return_counts=True
        )

        labels: Dict[str, str] = {
            "x": "確率",
            "y": "サンプリングされた回数",
        }
        df = pd.DataFrame({"x": unique_probabilities, "y": sample_counts})

        scatter_plot: Any = (
            alt.Chart(df)  # type: ignore
            .mark_circle(size=60)  # type: ignore
            .encode(
                alt.X(
                    f"x:Q",  # type: ignore
                    scale=alt.Scale(zero=False),  # type: ignore
                    axis=alt.Axis(title=labels.get("x")),  # type: ignore
                ),
                alt.Y(
                    f"y:Q",  # type: ignore
                    scale=alt.Scale(zero=False),  # type: ignore
                    axis=alt.Axis(title=labels.get("y")),  # type: ignore
                ),
                color=alt.value("steelblue"),  # type: ignore
            )
        )

        return scatter_plot


class GammaDist:
    @staticmethod
    def calculate_gamma_params(
        observed_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = observed_data.float().mean()

        var = observed_data.float().var()

        k = mean**2 / var
        theta = mean / var

        return k, theta
