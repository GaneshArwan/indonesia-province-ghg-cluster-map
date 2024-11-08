import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.header('Cluster Analysis of Greenhouse Gas Emissions by Province in Indonesia')
st.write("""
This application is part of a final project that aims to analyze and group Indonesian provinces based on 
their Greenhouse Gas (GHG) emission profiles using the K-Means Clustering method.

Greenhouse Gases (GHG) are gases that trap heat in the atmosphere, contributing to the greenhouse effect and 
global warming. The primary GHGs in Earth's atmosphere are water vapor, carbon dioxide (CO₂), methane (CH₄), 
nitrous oxide (N₂O), and ozone (O₃).

Indonesia is the 6th largest GHG emitter in the world as of 2021, contributing 1484660 Gg CO₂ eq-. Through this 
research, we hope to provide a better understanding of GHG emission patterns across various provinces in Indonesia.
""")

# Read the data
df_top_countries = pd.read_excel('./dataset/top_ghg_countries.xlsx')

# Create bar chart using plotly
fig = go.Figure(data=[
    go.Bar(
        x=df_top_countries['Emissions'],
        y=df_top_countries['Country'],
        orientation='h',
        marker=dict(
            color='#2E86C1',
            line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
        )
    )
])

fig.update_layout(
    title='Top 10 GHG Emitting Countries (Gg CO₂ eq-)',
    xaxis_title='GHG Emissions (Gg CO₂ eq-)',
    yaxis=dict(autorange="reversed"),
    height=500,
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)

st.write("""
In this analysis, two main methods are used:

1. K-Means Clustering is a machine learning algorithm that groups data into k clusters. The algorithm works by 
   finding cluster centers (centroids) that minimize the distance between data points within clusters through 
   an iterative process that continuously updates centroid positions until convergence is reached. 
   This method is crucial for our project as it helps identify distinct groups of provinces with similar GHG 
   emission profiles, enabling targeted policy-making and resource allocation.
   [Learn more](https://en.wikipedia.org/wiki/K-means_clustering)

2. Silhouette Coefficient is an evaluation metric that measures how well each object fits within its cluster. 
   Values range from -1 to 1, where values close to 1 indicate objects fit very well in their cluster, 
   values close to 0 indicate objects are on the border between two clusters, and values close to -1 indicate 
   objects might be placed in the wrong cluster. 
   In our project, the Silhouette Coefficient is used to validate the effectiveness of the clustering, ensuring 
   that the provinces are grouped optimally based on their GHG emissions.
   [Learn more](https://en.wikipedia.org/wiki/Silhouette_(clustering))

The analysis results will be visualized in an interactive map of Indonesia, where each province will be colored 
according to its resulting cluster. This map allows users to:
1. View province groupings based on GHG emission profiles
2. Explore detailed data for each province by clicking on the desired region
3. Compare emission patterns between provinces in the same or different clusters
""")


