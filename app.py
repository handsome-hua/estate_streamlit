import pickle
from PIL import Image
import streamlit as st
import streamlit.components.v1 as stc

# importing the smaller apps
from predict import run_random_forest_app
from eda_app import run_eda_app

html_temp = """
<div style="background-color:#FFFFFF;padding:10px;border-radius:10px">
    <h1 style="color:black;text-align:center;">Real Estate Price Prediction</h1>
    <h3 style="color:black;text-align:center;">Presented by : ZBH</h3>
</div>
"""

def main():
    stc.html(html_temp)

    menu = ["Home", "Data Analysis", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        # 美化的居中文字
        st.markdown("""
                <div style="text-align:center; font-size:24px; font-weight:bold; color:darkblue;">
                    一朝辞此地，四海遂为家
                </div>
                """, unsafe_allow_html=True)
    elif choice == "Data Analysis":
        run_eda_app()
    elif choice == "Prediction":
        run_random_forest_app()
main()