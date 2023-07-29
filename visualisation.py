import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Load the data
@st.cache
def load_data():
    projects = pd.read_csv('Verra_Projects.csv')
    ratings = pd.read_csv('Ratings.csv')
    return projects, ratings

projects, ratings = load_data()

# Merge the datasets
df = pd.merge(projects, ratings, left_on='ID', right_on='project_id', how='inner')

# Replace "Insufficient Information" with NaN
df.replace("N/A", np.nan, inplace=True)

# Sidebar filters
project_type_filter = st.sidebar.multiselect("Filter by Project Type", df['Project Type'].unique())
region_filter = st.sidebar.multiselect("Filter by Region", df['Region'].unique())
methodology_filter = st.sidebar.multiselect("Filter by Methodology", df['Methodology'].unique())
status_filter = st.sidebar.multiselect("Filter by Status", df['Status'].unique())
country_filter = st.sidebar.multiselect("Filter by Country/Area", df['Country/Area'].unique())
emission_filter = st.sidebar.slider("Filter by Estimated Annual Emission Reductions", min_value=0, max_value=int(df['Estimated Annual Emission Reductions'].max()), step=1000)
proponent_filter = st.sidebar.multiselect("Filter by Proponent", df['Proponent'].unique())

# Apply filters
filtered_df = df
if project_type_filter:
    filtered_df = filtered_df[filtered_df['Project Type'].isin(project_type_filter)]
if region_filter:
    filtered_df = filtered_df[filtered_df['Region'].isin(region_filter)]
if methodology_filter:
    filtered_df = filtered_df[filtered_df['Methodology'].isin(methodology_filter)]
if status_filter:
    filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]
if country_filter:
    filtered_df = filtered_df[filtered_df['Country/Area'].isin(country_filter)]
if emission_filter:
    filtered_df = filtered_df[filtered_df['Estimated Annual Emission Reductions'] >= emission_filter]
if proponent_filter:
    filtered_df = filtered_df[filtered_df['Proponent'].isin(proponent_filter)]

# Pie Chart of Most Popular Carbon Projects by Project Type
st.subheader("Most Popular Carbon Projects by Project Type")
project_type_counts = df['Project Type'].value_counts()
fig = px.pie(project_type_counts, values=project_type_counts.values, names=project_type_counts.index)
fig.update_layout(showlegend=False, legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='right', x=0.75))  # Add this line to adjust the legend size
st.plotly_chart(fig)

# Pie Chart of Regions with the Most Number of Carbon Projects
st.subheader("Regions with the Most Number of Carbon Projects")
region_counts = df['Region'].value_counts()
fig = px.pie(region_counts, values=region_counts.values, names=region_counts.index)
st.plotly_chart(fig)


# Calculate average scores by project developer
developer_avg_scores = filtered_df.groupby('Proponent')['overallScore'].mean().reset_index()
developer_avg_scores = developer_avg_scores.sort_values('overallScore', ascending=False)

# Count the number of projects in each status category for each project developer
status_counts = filtered_df.groupby(['Proponent', 'Status']).size().unstack(fill_value=0).reset_index()

# Merge the average scores table with the status counts table
developer_scores_counts = pd.merge(developer_avg_scores, status_counts, on='Proponent', how='left')

# Calculate the percentage of projects that are "On Hold" and "Inactive" for each project developer
try:
    developer_scores_counts['On Hold %'] = (developer_scores_counts['On Hold'] / developer_scores_counts['Registered']) * 100
except KeyError:
    developer_scores_counts['On Hold %'] = 0

try:
    developer_scores_counts['Inactive %'] = (developer_scores_counts['Inactive'] / developer_scores_counts['Registered']) * 100
except KeyError:
    developer_scores_counts['Inactive %'] = 0

# Display the updated table with the percentage of projects that are "On Hold" and "Inactive"
st.subheader("Project Developers with the Highest Average Score and Number of Projects")

# Create selectbox for column selection
sort_column = st.selectbox("Sort by", developer_scores_counts.columns)

# Create radio buttons for sort order
sort_order = st.radio("Sort order", ("Ascending", "Descending"))

# Sort the DataFrame based on user selection
if sort_order == "Ascending":
    developer_scores_counts = developer_scores_counts.sort_values(sort_column, ascending=True)
else:
    developer_scores_counts = developer_scores_counts.sort_values(sort_column, ascending=False)

# Display the sorted table
st.table(developer_scores_counts)


# Calculate average scores by region
region_avg_scores = df.groupby('Region')[['projectDetails_score', 'safeguards_score', 'applicabilityOfMethodology_score', 'projectBoundary_score', 'baseline_score', 'additionality_score', 'emissionReductions_score', 'monitoringPlan_score']].mean().reset_index()

# Reshape the dataframe to have separate rows for each score category
region_avg_scores = region_avg_scores.melt(id_vars='Region', var_name='Score Category', value_name='Average Score')

# Bar Chart of Average Scores by Region and Score Category
st.subheader("Average Scores by Region and Score Category")
fig = px.bar(region_avg_scores, x='Region', y='Average Score', color='Score Category', barmode='group')
st.plotly_chart(fig)

# Heatmap of Ratings
st.subheader("Heatmap of Ratings")
ratings_heatmap = df[['projectDetails_score', 'safeguards_score', 'applicabilityOfMethodology_score', 'projectBoundary_score', 'baseline_score', 'additionality_score', 'emissionReductions_score', 'monitoringPlan_score']]
ratings_heatmap = ratings_heatmap.astype(float)  # Convert the columns to float
fig = px.imshow(ratings_heatmap.corr(), color_continuous_scale='RdBu_r')
st.plotly_chart(fig)

# Correlation Matrix Plot
st.subheader("Correlation Matrix Plot")
fig = px.scatter_matrix(df, dimensions=['projectDetails_score', 'safeguards_score', 'applicabilityOfMethodology_score', 'projectBoundary_score', 'baseline_score', 'additionality_score', 'emissionReductions_score', 'monitoringPlan_score'], color='overallScore')
st.plotly_chart(fig)

# Histogram of Overall Scores
st.subheader("Histogram of Overall Scores")
fig = px.histogram(df, x='overallScore')
st.plotly_chart(fig)

# Boxplot of Overall Scores by Project Type
st.subheader("Boxplot of Overall Scores by Project Type")
fig = px.box(df, x='Project Type', y='overallScore')
st.plotly_chart(fig)

# Scatter plot of estimated annual emission reductions and overall score
st.subheader("Scatter plot of estimated annual emission reductions and overall score")
fig = px.scatter(df, x='Estimated Annual Emission Reductions', y='overallScore')
st.plotly_chart(fig)

# Bar Chart of Average Category Scores
st.subheader("Bar Chart of Average Category Scores")
average_scores = df[['projectDetails_score', 'safeguards_score', 'applicabilityOfMethodology_score', 'projectBoundary_score', 'baseline_score', 'additionality_score', 'emissionReductions_score', 'monitoringPlan_score']].mean().reset_index()
average_scores.columns = ['Category', 'Average Score']
fig = px.bar(average_scores, x='Category', y='Average Score')
st.plotly_chart(fig)
