import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
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

# Predictions Dashboard
with tab1:
    # Filter Data 
    st.subheader('Filter')
    values = st.slider(label, min_value=0, max_value=100, value=(50,100), step=5, format="%d%%")
    
    # Apply filter
    cond1 = df[col] >= values[0]/100
    cond2 = df[col] <= values[1]/100
    filtered_df = df[cond1&cond2]
    # Format table to print
    filtered_df['EmployeeID'] = filtered_df['EmployeeID'].astype(str)
    filtered_df = filtered_df.set_index('EmployeeID')
    filtered_df[label] = round(filtered_df[col]*100,1)

    # Plot histogram
    fig = ff.create_distplot([df[col]], group_labels=[label], bin_size=0.05, show_rug=False, colors=['lightblue'])
    fig.add_vrect(x0=values[0]/100, x1=values[1]/100, line_width=0, fillcolor="orange", opacity=0.2)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Add historgram to streamlit
        st.subheader('Attrition Propesity Distribution')
        n = filtered_df.shape[0]
        total = df.shape[0]
        st.write(f'**{n}** of the **{total}** employees have attrition propensity between **{values[0]}%** and **{values[1]}%**.')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    with col2:
        # Print table with top 100 at risk employees
        st.subheader('Employees at Risk')
        st.write('Shows top 100 employees at risk of leaving within the selected attrition propensity range filter.')
        # Center the dataframe
        with st.columns([1,3,1])[1]:
            st.dataframe(filtered_df[[label]].sort_values(label, ascending=False).head(100))

# Employee Risk
with tab2:
    # Get list of employees to add to Selection Box
    employee_ids = df['EmployeeID'].unique()
    employee_id = st.selectbox('Select Employee ID', employee_ids)

    # Filter data for selected employee
    cond3 = df['EmployeeID'] == employee_id
    employee_data = df[cond3]
    employee_data.set_index('EmployeeID', inplace=True)
    st.write(f'Employee **{employee_id}** has a propensity to leave of:', "**{:.2%}**".format(employee_data[col].iloc[0]))


    # Extract model and preprocessor from model pipeline
    preprocessor = model_pipeline[0]
    model = model_pipeline[-1]

    # Calculate feature impact
    x_t = preprocessor.transform(employee_data[x_cols])
    coefs = model.coef_[0]
    feature_impact = x_t * coefs
    
    # Transform feature impact to pandas Dataframe
    feature_names = preprocessor.get_feature_names_out()
    feature_impact_df = pd.DataFrame(feature_impact.T,
        columns=['feature_impact'],
        index=feature_names,
    )

    st.header('Top attributes increasing employee\'s propensity to leave')

    col1, col2 = st.columns([2, 1])

    # TABLE - Show feature attributes
    with col2:
        st.subheader('Attributes')

        # Top 10 features increasing propensity to leave
        orig_features = [feature.split('__')[1].split('_')[0] for feature in feature_impact_df.index]
        feature_impact_df['orig_features'] = orig_features
        grouped_df = pd.DataFrame(feature_impact_df.groupby('orig_features').sum()['feature_impact'])
        grouped_df['abs_feature_impact'] = np.abs(grouped_df['feature_impact'])
        top_10_features = grouped_df.sort_values('abs_feature_impact',ascending=False).head(10)
        
        st.table(employee_data[top_10_features.index].T)

    # CHART - Show relative impact on propensity score
    with col1:
        st.subheader('Impact on propensity to leave')
        st.write('Positive values indicates the feature increases the attrition propensity and the negative values indicates a decrease in attrition propensity.')
        
        # Format data for plotly chart
        top_10_features = top_10_features.reset_index()
        top_10_features['Positive Impact'] = top_10_features['feature_impact'] > 0
        top_10_features.rename({'orig_features':'Attribute', 'feature_impact': 'Feature Impact'}, axis=1, inplace=True)

        # https://plotly.com/python/discrete-color/
        discrete_colors = [ px.colors.qualitative.Plotly[index] for index in [2,1] ]
        fig = px.bar(top_10_features, x="Feature Impact", y="Attribute", color='Positive Impact', 
            orientation='h', height=600, color_discrete_sequence=discrete_colors)
        # Select order of the bars in the chart
        order_list = list(top_10_features['Attribute'].values)
        order_list.reverse()
        fig.update_yaxes(categoryorder='array', categoryarray= order_list)
        fig.update_layout(showlegend=False)

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


