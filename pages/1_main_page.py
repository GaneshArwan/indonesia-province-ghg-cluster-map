import streamlit as st

st.header('Cluster Analysis of Greenhouse Gas Emissions by Province in Indonesia')
st.write("""
This application is part of a thesis research that aims to analyze and group Indonesian provinces based on 
their Greenhouse Gas (GHG) emission profiles using the K-Means Clustering method.

Indonesia is the 6th largest GHG emitter in the world as of 2021, contributing 1.48 Gt CO₂ eq-. Through this 
research, we hope to provide a better understanding of GHG emission patterns across various provinces in Indonesia.

In this analysis, two main methods are used:

K-Means Clustering is a machine learning algorithm that groups data into k clusters. The algorithm works by 
finding cluster centers (centroids) that minimize the distance between data points within clusters through 
an iterative process that continuously updates centroid positions until convergence is reached.

Silhouette Coefficient is an evaluation metric that measures how well each object fits within its cluster. 
Values range from -1 to 1, where values close to 1 indicate objects fit very well in their cluster, 
values close to 0 indicate objects are on the border between two clusters, and values close to -1 indicate 
objects might be placed in the wrong cluster.

The analysis results will be visualized in an interactive map of Indonesia, where each province will be colored 
according to its resulting cluster. This map allows users to:
- View province groupings based on GHG emission profiles
- Explore detailed data for each province by clicking on the desired region
- Compare emission patterns between provinces in the same or different clusters

For more information, please visit:
- [K-means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [Silhouette Coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering))
""")


