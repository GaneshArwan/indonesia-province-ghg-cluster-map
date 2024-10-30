import streamlit as st

st.header('About Me')

st.write("""

This application was developed as part of my final project for my college degree. It helps visualize and analyze greenhouse gas (GHG) emissions across Indonesian provinces using K-means clustering to identify patterns and group provinces based on their emission levels.

## Data Source

The example dataset used in this application is sourced from [SIGNSMART (Sistem Inventarisasi GRK Nasional)](https://signsmart.menlhk.go.id/v2.1/), Indonesia's National GHG Inventory System managed by the Ministry of Environment and Forestry. This platform provides official greenhouse gas emissions data across Indonesian provinces.

## Data Privacy

Your data security is important. All uploaded datasets are processed in-memory and are not stored on our servers. The source code is available for review on [GitHub](https://github.com/GaneshArwan/indonesia-province-ghg-cluster-map).

## Contact & Support

I welcome your feedback to improve this application. For questions, suggestions, or issues:
- Submit an issue on [GitHub](https://github.com/GaneshArwan/indonesia-province-ghg-cluster-map/issues)
- Participate in GitHub discussions

Thank you for using this application!
""")
