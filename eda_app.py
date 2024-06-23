import pandas as pd
from PIL import Image
import streamlit as st


def run_eda_app():
    st.subheader("Real Estate : Data Analysis")

    submenu = st.sidebar.selectbox("Submenu", ["Descriptive", "Plots"])
    df = pd.read_csv("data_last.csv")
    if submenu == "Descriptive":
        img1 = Image.open("Image/Real_Estate.png")
        st.image(img1)

        with st.expander("Dataset"):
            st.dataframe(df)

        with st.expander("Data Types"):
            st.dataframe(df.dtypes)

        with st.expander("Data Summary"):
            st.dataframe(df.describe())

        # with st.expander("Location Distribution"):
        #     st.dataframe(df["Region"].value_counts().head(30))

    elif submenu == "Plots":

        with st.expander("3d_surface_plot_improved"):
            img2 = Image.open("Image/3d_surface_plot_improved.png")
            st.image(img2)

        with st.expander("contour_plot_improved"):
            img3 = Image.open("Image/contour_plot_improved.png")
            st.image(img3)

        with st.expander("floor_level2price.png"):
            img4 = Image.open("Image/floor_level2price.png")
            st.image(img4)

        with st.expander("long_lat2price"):
            img5 = Image.open("Image/long_lat2price.png")
            st.image(img5)

        with st.expander("map"):
            img6 = Image.open("Image/map.png")
            st.image(img6)

        with st.expander("squre2price.png"):
            img7 = Image.open("Image/squre2price.png")
            st.image(img7)