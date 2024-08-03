import streamlit as st
import requests
from streamlit_lottie import st_lottie
import joblib
import numpy as np
import joblib

#Page title
st.set_page_config(page_title="Breast Cancer Detection")

#Checking the page animation
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#Transforming data from column to row => .reshape
def prepareData(radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,
                symmetry_mean,fractal_dimension_mean,
                                    radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,
                                    symmetry_se,fractal_dimension_se,
                                    radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,
                                    symmetry_worst,fractal_dimension_worst):


    a =[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,
                symmetry_mean,fractal_dimension_mean,
                                    radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,
                                    symmetry_se,fractal_dimension_se,
                                    radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,
                                    symmetry_worst,fractal_dimension_worst]
    sample = np.array(a).reshape(-1,len(a))
    scaler_loaded = joblib.load(open("breast_cancer_scaler", 'rb'))
    sample = scaler_loaded.transform(sample)
    return sample

#Loading the model
loaded_model = joblib.load(open("breast_cancer_project", 'rb'))

#Loading the animation
#lottie_link = "https://assets9.lottiefiles.com/packages/lf20_4p5C1p.json"
#animation = load_lottie(lottie_link)


st.write("# Breast Cancer Model Deployment")

#with st.container():
    #st_lottie( , speed=1 , height = 200 , key="initial")

st.write("---")
st.subheader("Enter your Details")


with st.container():

    left_column, right_column = st.columns(2)

    with left_column:
        radius_mean = st.number_input("Radius mean: ",format="%.6f", step = 0.000001,min_value=0.0)
        texture_mean = st.number_input("Texture mean: ",format="%.6f", step = 0.000001,min_value=0.0)
        perimeter_mean = st.number_input("Perimeter mean: ",format="%.6f", step = 0.000001,min_value=0.0)
        area_mean = st.number_input("Area mean: ",format="%.6f", step = 0.000001,min_value=0.0)
        smoothness_mean = st.number_input("Smoothness mean: ",format="%.6f", step = 0.000001,min_value=0.0)
        compactness_mean = st.number_input("Compactness mean: ",format="%.6f", step = 0.000001,min_value=0.0)
        concavity_mean = st.number_input("Concavity mean: ",format="%.6f", step = 0.000001,min_value=0.0)
        concave_points_mean = st.number_input("Concave Points mean: ",format="%.6f", step = 0.000001,min_value=0.0)
        symmetry_mean = st.number_input("Symmetry mean: ",format="%.6f", step = 0.000001,min_value=0.0)
        fractal_dimension_mean = st.number_input("Fractal Dimension mean: ",format="%.6f", step = 0.000001,min_value=0.0)

        radius_se = st.number_input("Radius SE: ",format="%.6f", step = 0.000001,min_value=0.0)
        texture_se = st.number_input("Texture SE: ",format="%.6f", step = 0.000001,min_value=0.0)
        perimeter_se = st.number_input("Perimeter SE: ",format="%.6f", step = 0.000001,min_value=0.0)
        area_se = st.number_input("Area SE: ",format="%.6f", step = 0.000001,min_value=0.0)
        smoothness_se = st.number_input("Smoothness SE: ",format="%.6f", step = 0.000001,min_value=0.0)

    with right_column:
        compactness_se = st.number_input("Compactness SE: ",format="%.6f",step = 0.000001,min_value=0.0)
        concavity_se = st.number_input("Concavity SE: ",format="%.6f",step = 0.000001,min_value=0.0)
        concave_points_se = st.number_input("Concave Points SE: ",format="%.6f",step = 0.000001,min_value=0.0)
        symmetry_se = st.number_input("Symmetry SE: ",format="%.6f",step = 0.000001,min_value=0.0)
        fractal_dimension_se = st.number_input("Fractal Dimension SE: ",format="%.6f",step = 0.000001,min_value=0.0)

        radius_worst = st.number_input("Radius worst: ",format="%.6f",step = 0.000001,min_value=0.0)
        texture_worst = st.number_input("Texture worst: ",format="%.6f",step = 0.000001,min_value=0.0)
        perimeter_worst = st.number_input("Perimeter worst: ",format="%.6f",step = 0.000001,min_value=0.0)
        area_worst = st.number_input("Area worst: ",format="%.6f",step = 0.000001,min_value=0.0)
        smoothness_worst = st.number_input("Smoothness worst: ",format="%.6f",step = 0.000001,min_value=0.0)
        compactness_worst = st.number_input("Compactness worst: ",format="%.6f",step = 0.000001,min_value=0.0)
        concavity_worst = st.number_input("Concavity worst: ",format="%.6f",step = 0.000001,min_value=0.0)
        concave_points_worst = st.number_input("Concave Points worst: ",format="%.6f",step = 0.000001,min_value=0.0)
        symmetry_worst = st.number_input("Symmetry worst: ",format="%.6f",step = 0.000001,min_value=0.0)
        fractal_dimension_worst = st.number_input("Fractal Dimension worst: ",format="%.6f",step = 0.000001,min_value=0.0)

        #Button to load the model when clicked
        if st.button("Show diagnosis"):
            sample = prepareData(radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,
                                symmetry_mean,fractal_dimension_mean,
                                radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,
                                symmetry_se,fractal_dimension_se,
                                radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,
                                symmetry_worst,fractal_dimension_worst
                            )
            predy = loaded_model.predict(sample)
            if predy == 1: #Malignant
                st.write("The tumor is Malignant.")
            else: # Benign
                st.write("The tumor is Benign.")


