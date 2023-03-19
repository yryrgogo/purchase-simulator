import streamlit as st
import torch
from dashboard.form import create_form
from simulation.user_purchase_per_store import UserPurchasePerStoreSimulator
from simulation.utils import get_weekday_name


def get_data() -> torch.Tensor:
    observed_data = torch.tensor([0.1, 0.5, 1, 0])
    return observed_data


def create_dashboard():
    started, user_id = create_form()

    if started:
        if user_id is None:
            raise ValueError("user_id is None")
        simulator = UserPurchasePerStoreSimulator(user_id)
        simulator.simulate()

        for weekday, purchase_count in enumerate(simulator.simulated_weekday_purchase_counts):
            st.write(f"{get_weekday_name(weekday)}曜日の購買数: {purchase_count}")  # type: ignore

        # st.altair_chart(chart, use_container_width=True)


# def create_chart(samples: List[float]) -> Any:
#     chart = PosteriorResult(samples).create_scatter_plot()

#     return chart
