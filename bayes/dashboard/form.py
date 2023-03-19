import streamlit as st


def create_form() -> tuple[bool, str | None]:
    st.title("ユーザー購買シミュレータ")

    submitted = False
    with st.form(key="simulation_form"):
        user_id, selected =  add_user_id_selectbox()

        if selected:
            submitted = st.form_submit_button("シミュレーション実行")
            if submitted:
                st.write("シミュレーション開始")  # type: ignore
        else:
            st.form_submit_button("シミュレーション実行", on_click=None)

    return submitted, user_id

def add_user_id_selectbox():
    st.title("選択フォーム")
    place_holder = "選択してください"
    option = st.selectbox("番号を選択してください", [place_holder, "123", "456", "789"])
    selected = option != place_holder
    if selected:
        st.write(f"選択された番号は {option} です。")  # type: ignore
    return option, selected
