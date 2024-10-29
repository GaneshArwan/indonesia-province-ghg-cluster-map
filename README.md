# Exploring K-Means Clustering in Scikit-learn with Streamlit

Try the app here: [K-Means Clustering for Indonesian GHG Emissions](your-streamlit-link-here)

![page preview](./miscellaneous/indonesia_map.png)

This app lets you explore Scikit-learn's K-Means clustering algorithm using Indonesian greenhouse gas emissions data. You can use your own dataset as a CSV or Excel file to perform the clustering, or you can use the provided example dataset containing greenhouse gas emissions data across Indonesian provinces. If you choose to use your own data, rest assured that it won't be stored anywhere.

You can choose how many dimensions to use for clustering by selecting specific years of data, or include all years for a comprehensive analysis. The application visualizes the clustering results on an interactive map of Indonesia, helping identify patterns and similarities in emissions across different provinces.

# Create and activate virtual environment
```
python -m venv venv
```
# On Unix/MacOS
```
source venv/bin/activate  # On Unix/MacOS
```
# On Windows
```
.\venv\Scripts\activate  # On Windows
```

# Install dependencies
```
pip install -r requirements.txt
```

# Technical Implementation
- K-Means Clustering for unsupervised learning
- Integration with Scikit-learn ML library
- Streamlit web application development
- Silhouette analysis for optimal cluster determination
- Interactive geospatial visualization using AmCharts
- Data preprocessing and ETL pipeline
- Custom data template generation

Lottie animations from [LottieFiles](https://lottiefiles.com/free-animation/statistics-z17P9Q8377).