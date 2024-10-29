import streamlit as st

st.header('About This Project')

st.write("""
# Indonesia Province GHG Clustering App

This application was developed as part of my final project for my college degree. It helps visualize and analyze greenhouse gas (GHG) emissions across Indonesian provinces using K-means clustering to identify patterns and group provinces based on their emission levels.

## Key Features

- Upload custom datasets or use the provided example dataset
- Interactive data preview and preprocessing tools
- K-means clustering to group provinces by emission levels with map visualization
- Silhouette score analysis for optimal cluster determination
- Detailed province-level emission data tooltips

## Data Privacy

Your data security is important. All uploaded datasets are processed in-memory and are not stored on our servers. The source code is available for review on [GitHub](https://github.com/vuxvix/indonesia-province-ghg-cluster-map).

## Contact & Support

I welcome your feedback to improve this application. For questions, suggestions, or issues:
- Submit an issue on [GitHub](https://github.com/vuxvix/indonesia-province-ghg-cluster-map/issues)
- Participate in GitHub discussions

Thank you for using this application!
""")
