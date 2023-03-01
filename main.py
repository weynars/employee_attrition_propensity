import streamlit as st
import pandas as pd
import numpy as np
import joblib

# import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(layout="wide")
st.title('Healthcare Attrition Propensity')


filename = './data/healthcare_data.csv'

x_cols = ['BusinessTravel',
    'Department',
    'Education',
    'EducationField',
    'EnvironmentSatisfaction',
    'Gender',
    'JobInvolvement',
    'JobLevel',
    'JobRole',
    'JobSatisfaction',
    'MaritalStatus',
    'OverTime',
    'PerformanceRating',
    'RelationshipSatisfaction',
    'Shift',
    'WorkLifeBalance',
    'Age',
    'DailyRate',
    'DistanceFromHome',
    'HourlyRate',
    'MonthlyIncome',
    'MonthlyRate',
    'NumCompaniesWorked',
    'PercentSalaryHike',
    'TotalWorkingYears',
    'TrainingTimesLastYear',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager']

@st.cache_data
def fetch_data(filename):
    df = pd.read_csv(filename)
    return df

@st.cache_data
def load_model():
    model_pipeline = joblib.load('model.sav')
    return model_pipeline

@st.cache_data
def model_inference(_model, df, cols):
    x = df[cols]
    y_prob = _model.predict_proba(x)
    return y_prob


# Load data
df = fetch_data(filename)
# Load model
model_pipeline = load_model()
# Make inference on data
y_prob = model_inference(_model=model_pipeline, df=df, cols=x_cols)

# Add inference to dataset
df['y_prob'] = y_prob[:,1]

# Columns to use in the chart
col = 'y_prob'
label = 'Attrition Propensity'

tab1, tab2 = st.tabs(["Predictions Dashboard", "Employee Risk"])

with tab1:
    st.subheader('Filter')
    values = st.slider(label, min_value=0, max_value=100, value=(50,100), step=5, format="%d%%")

    # Filter Data 
    
    # Apply filter
    cond1 = df[col] >= values[0]/100
    cond2 = df[col] <= values[1]/100
    filtered_df = df[cond1&cond2]
    # Format table to print
    filtered_df['EmployeeID'] = filtered_df['EmployeeID'].astype(str)
    filtered_df = filtered_df.set_index('EmployeeID')
    filtered_df[label] = filtered_df[col]

    # Plot histogram
    fig = ff.create_distplot([df[col]], group_labels=[label], bin_size=0.05, show_rug=False, colors=['lightblue'])
    fig.add_vrect(x0=values[0]/100, x1=values[1]/100, line_width=0, fillcolor="orange", opacity=0.2)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Add historgram to streamlit
        st.subheader('Attrition Propesity Distribution')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    with col2:
        # Print table with top 100 at risk employees
        st.subheader('Employees at Risk:')
        st.write('Shows top 100 employees at risk of leaving within the selected attritiob propensity range filter.')
        # Center the dataframe
        with st.columns([1,3,1])[1]:
            st.dataframe(filtered_df[[label]].sort_values(label, ascending=False).head(100))

with tab2:
    st.write('EmployeeID')

