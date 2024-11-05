import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import streamlit_shadcn_ui as ui
import streamlit.components.v1 as components
import json
import math
from matplotlib.colors import ListedColormap

# Add this line to define numeric types
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# Move checkbox to sidebar
use_example_file = st.sidebar.checkbox("Use example file", False, help="Use in-built example file to demo the app")

# Add notice before file upload
st.info("""
    **Before uploading your file, please ensure:**
    - Your data is in excel file format
    - I provide a template file to download below
    - Your data follows the template format
    - Provinces id and name is not modified
    - Values are in Gigagram (Gg) units because it's the unit used in Indonesia
    - Numbers should be written without dots for thousands (e.g., write 1000 not 1.000)
    - Use comma or dot for decimal points if needed (e.g., 1000,5 or 1.000.5)
""")

# Keep file upload in main area
st.header("Data Upload and Clustering")
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

# Add download button for example file
example_path = "./dataset/dataset_emission_province_by_sectors_in_indonesia.xlsx"
template_path = "./dataset/template.xlsx"
with open(template_path, "rb") as file:
    st.download_button(
        label="Download template file",
        data=file,
        file_name="template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if use_example_file:
    uploaded_file = example_path
    file_name = uploaded_file
else:
    file_name = uploaded_file.name if uploaded_file is not None else None

df = None  # Initialize df as None
def preprocess_data(df):
    # Assuming the first column is Id and the second is Province
    numeric_columns = df.columns[2:]
        
    # Replace "0,00" and "0" with NaN in numeric columns
    df[numeric_columns] = df[numeric_columns].replace({"0,00": np.nan, "0": np.nan})

    #Change the values to float and replace , with .
    df[numeric_columns] = df[numeric_columns].replace(regex={',': '.'}).astype(float)
    
    # Fill missing values in numeric columns row-wise
    df[numeric_columns] = df[numeric_columns].ffill(axis = 1).bfill(axis = 1)
    
    # Replace remaining NaN values with 0
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Remove columns with empty or no headers
    df = df.dropna(axis=1, how='all')  # Remove columns where all values are NaN
    df = df.loc[:, df.columns != '']  # Remove columns with empty string as header
    
    # Convert object columns to string
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    
    return df

def display_data(df):
    st.dataframe(df)
    st.write(df.describe())
    st.write(f"#### Our dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    st.write(check_nunique_missing(df))

def check_nunique_missing(df):
    check_list = []
    for col in df.columns:
        dtypes = df[col].dtypes
        nunique = df[col].nunique()
        not_na = df[col].notna().sum()
        sum_na = df[col].isna().sum()
        check_list.append([col, dtypes, nunique, not_na, sum_na])
    df_check = pd.DataFrame(check_list, columns=['column', 'dtypes', 'nunique', 'not_na', 'sum_na'])
    return df_check

def perform_clustering(df, columns_for_model):
    X = df.loc[:, columns_for_model].values
    
    max_clusters = 10
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=101)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because we start from 2 clusters
    
    with st.expander("Show Silhouette Score Visualization"):
        fig, ax = plt.subplots()
        ax.plot(range(2, max_clusters + 1), silhouette_scores)
        ax.set_title('Silhouette Score Evaluation')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Silhouette Score')
        st.pyplot(fig)
    
    top_3_scores = sorted([(i+2, score) for i, score in enumerate(silhouette_scores)], key=lambda x: x[1], reverse=True)[:3]
    st.write("### Top 3 highest scoring province groupings (clusters / k):")
    for cluster, score in top_3_scores:
        st.write(f"#### Group into {cluster} clusters (Score: {score:.4f} out of 1.0)")
    
    num_clusters = st.sidebar.slider("Decide on the final number of clusters", 2, 10, optimal_clusters)
    kmeansmodel = KMeans(n_clusters=num_clusters, random_state=101)
    y_kmeans = kmeansmodel.fit_predict(X)
    
    # Add cluster labels to the dataframe
    df['Cluster'] = y_kmeans + 1  # Adding 1 to make clusters 1-based instead of 0-based
    
    # Add data preview with selected columns and cluster
    with st.expander("Show Data of Province with Cluster"):
        preview_df = df[['PROVINCE', 'Cluster']]  # Select Province and Cluster columns
        st.dataframe(preview_df)
    
    # Add a section for the map
    st.header("Indonesia Province Map Visualization")
    
    # Add Gg explanation
    st.info("""
        **Note on units:** Values are shown in Gg (Gigagram)
        - 1 Gg = 1,000,000,000 grams (1 billion grams)
        - This unit is commonly used for reporting greenhouse gas emissions in Indonesia
    """)
    
    if df is not None and selected_sheets and columns_for_model:
        display_province_map_2d(df, columns_for_model)
    else:
        st.warning("Please load data and select columns for clustering first")

def display_province_map_2d(df=None, selected_columns=None):
    # Define province codes dictionary
    province_codes = {
        "ACEH": "ID-AC",
        "SUMATERA UTARA": "ID-SU",
        "SUMATERA BARAT": "ID-SB",
        "RIAU": "ID-RI",
        "JAMBI": "ID-JA",
        "SUMATERA SELATAN": "ID-SS",
        "BENGKULU": "ID-BE",
        "LAMPUNG": "ID-LA",
        "BANGKA BELITUNG": "ID-BB",
        "KEPULAUAN RIAU": "ID-KR",
        "DKI JAKARTA": "ID-JK",
        "JAWA BARAT": "ID-JB",
        "JAWA TENGAH": "ID-JT",
        "DI YOGYAKARTA": "ID-YO",
        "JAWA TIMUR": "ID-JI",
        "BANTEN": "ID-BT",
        "BALI": "ID-BA",
        "NUSA TENGGARA BARAT": "ID-NB",
        "NUSA TENGGARA TIMUR": "ID-NT",
        "KALIMANTAN BARAT": "ID-KB",
        "KALIMANTAN TENGAH": "ID-KT",
        "KALIMANTAN SELATAN": "ID-KS",
        "KALIMANTAN TIMUR": "ID-KI",
        "KALIMANTAN UTARA": "ID-KU",
        "SULAWESI UTARA": "ID-SA",
        "SULAWESI TENGAH": "ID-ST",
        "SULAWESI SELATAN": "ID-SN",
        "SULAWESI TENGGARA": "ID-SG",
        "GORONTALO": "ID-GO",
        "SULAWESI BARAT": "ID-SR",
        "MALUKU": "ID-MA",
        "MALUKU UTARA": "ID-MU",
        "PAPUA": "ID-PA",
        "PAPUA BARAT": "ID-PB",
    }

    # Create legend data outside the map
    legend_data = {
        "Cluster 1": "#FF9999",  # Light red
        "Cluster 2": "#99FF99",  # Light green
        "Cluster 3": "#9999FF",  # Light blue
        "Cluster 4": "#FFFF99",  # Light yellow
        "Cluster 5": "#FF99FF",  # Light purple
        "Cluster 6": "#99FFFF",  # Light cyan
        "Cluster 7": "#FFB366",  # Light orange
        "Cluster 8": "#B366FF",  # Light violet
        "Cluster 9": "#66FFB3",  # Light mint
        "Cluster 10": "#FF66B3"  # Light pink
    }

    with st.spinner('Loading AmCharts map...'):
        if df is not None and selected_columns is not None:
            # Sort the selected columns
            selected_columns = sorted(selected_columns)
            sample_data = []
            
            for idx, row in df.iterrows():
                province_name = row.iloc[1].upper().strip()
                province_id = province_codes.get(province_name, "")
                
                if province_id:
                    province_data = {
                        "id": province_id,
                        "name": row.iloc[1],
                        "value": 0,  # Default value
                        "cluster": int(row['Cluster'])  # Add cluster information
                    }
                    for col in selected_columns:
                        try:
                            province_data[str(col)] = float(row[col])
                        except (ValueError, TypeError):
                            province_data[str(col)] = 0
                    
                    sample_data.append(province_data)
                else:
                    st.warning(f"Province not found in mapping: {province_name}")

        # Update tooltip to include cluster and handle many columns
        if len(selected_columns) > 5:
            visible_years = selected_columns[:5]
            tooltip_visible = "\n".join([f"{year}: {{{year}}} Gg" for year in visible_years])
            
            tooltip_text = f"""[bold]{{name}}[/]
Cluster: {{cluster}}
{tooltip_visible}
[bold][#666666]Click to show more...[/][/]"""
        else:
            tooltip_text = f"""[bold]{{name}}[/]
Cluster: {{cluster}}
{"\n".join([f"{year}: {{{year}}} Gg" for year in selected_columns])}"""

        map_html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Indonesia Map</title>
            <script src="https://cdn.amcharts.com/lib/5/index.js"></script>
            <script src="https://cdn.amcharts.com/lib/5/map.js"></script>
            <script src="https://cdn.amcharts.com/lib/5/geodata/indonesiaHigh.js"></script>
            <script src="https://cdn.amcharts.com/lib/5/themes/Animated.js"></script>
            <script src="https://cdn.amcharts.com/lib/5/locales/id_ID.js"></script>
            <style>
                #chartdiv {
                    width: 100%;
                    height: 800px;
                }
                .modal {
                    display: none;
                    position: fixed;
                    z-index: 1000;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(0,0,0,0.4);
                }
                .modal-content {
                    background-color: #fefefe;
                    margin: 15% auto;
                    padding: 20px;
                    border: 1px solid #888;
                    width: 80%;
                    max-width: 500px;
                    border-radius: 8px;
                    position: relative;
                }
                .close {
                    color: #aaa;
                    float: right;
                    font-size: 28px;
                    font-weight: bold;
                    cursor: pointer;
                }
                .close:hover {
                    color: black;
                }
                .modal-title {
                    font-size: 1.5em;
                    margin-bottom: 15px;
                    color: #333;
                }
                .modal-data {
                    max-height: 400px;
                    overflow-y: auto;
                }
                .data-row {
                    display: flex;
                    justify-content: space-between;
                    padding: 8px 0;
                    border-bottom: 1px solid #eee;
                }
                .data-label {
                    font-weight: bold;
                    color: #555;
                }
            </style>
        </head>
        <body>
            <div id="chartdiv"></div>
            <div id="dataModal" class="modal">
                <div class="modal-content">
                    <span class="close">&times;</span>
                    <div class="modal-title"></div>
                    <div class="modal-data"></div>
                </div>
            </div>
            <script>
            am5.ready(function() {
                var root = am5.Root.new("chartdiv");
                root.setThemes([am5themes_Animated.new(root)]);
                
                var chart = root.container.children.push(
                    am5map.MapChart.new(root, {
                        panX: "translateX",
                        panY: "translateY",
                        projection: am5map.geoMercator(),
                        homeGeoPoint: { longitude: 118, latitude: -2 }
                    })
                );

                // Filter out unwanted regions
                var filteredGeoData = Object.assign({}, am5geodata_indonesiaHigh);
                filteredGeoData.features = filteredGeoData.features.filter(function(feature) {
                    return !["MY-12", "MY-13", "TL", "BN"].includes(feature.properties.id);  // Added "BN" for Brunei
                });

                // Define colors for different clusters
                var colors = {
                    1: am5.color(0xFF9999),  // Light red
                    2: am5.color(0x99FF99),  // Light green
                    3: am5.color(0x9999FF),  // Light blue
                    4: am5.color(0xFFFF99),  // Light yellow
                    5: am5.color(0xFF99FF),  // Light purple
                    6: am5.color(0x99FFFF),  // Light cyan
                    7: am5.color(0xFFB366),  // Light orange
                    8: am5.color(0xB366FF),  // Light violet
                    9: am5.color(0x66FFB3),  // Light mint
                    10: am5.color(0xFF66B3)  // Light pink
                };

                var polygonSeries = chart.series.push(
                    am5map.MapPolygonSeries.new(root, {
                        geoJSON: filteredGeoData,
                        valueField: "value",
                        calculateAggregates: true
                    })
                );

                // Get modal elements
                var modal = document.getElementById("dataModal");
                var span = document.getElementsByClassName("close")[0];

                // Close modal when clicking (x)
                span.onclick = function() {
                    modal.style.display = "none";
                }

                // Close modal when clicking outside
                window.onclick = function(event) {
                    if (event.target == modal) {
                        modal.style.display = "none";
                    }
                }

                polygonSeries.mapPolygons.template.setAll({
                    tooltipText: ''' + json.dumps(tooltip_text) + ''',
                    interactive: true,
                    fill: am5.color(0xBBBBBB),
                    strokeWidth: 2,
                    stroke: am5.color(0x000000),
                    strokeOpacity: 0.5
                });

                // Add click handler for modal
                polygonSeries.mapPolygons.template.events.on("click", function(ev) {
                    var dataContext = ev.target.dataItem.dataContext;
                    
                    // Set modal title
                    document.querySelector(".modal-title").innerHTML = 
                        `<strong>${dataContext.name}</strong> (Cluster ${dataContext.cluster})`;
                    
                    // Build modal content
                    var modalContent = "";
                    ''' + json.dumps(selected_columns) + '''.forEach(function(year) {
                        modalContent += `
                            <div class="data-row">
                                <span class="data-label">${year}:</span>
                                <span>${dataContext[year].toLocaleString('en-US', {
                                    minimumFractionDigits: 2,
                                    maximumFractionDigits: 2
                                })} Gg</span>
                            </div>
                        `;
                    });
                    
                    document.querySelector(".modal-data").innerHTML = modalContent;
                    
                    // Show modal
                    modal.style.display = "block";
                });

                polygonSeries.mapPolygons.template.adapters.add("fill", function(fill, target) {
                    if (target.dataItem) {
                        var cluster = target.dataItem.dataContext.cluster;
                        return colors[cluster] || fill;
                    }
                    return fill;
                });

                polygonSeries.mapPolygons.template.states.create("hover", {
                    fillOpacity: 0.6,
                    strokeWidth: 3
                });

                polygonSeries.data.setAll(''' + json.dumps(sample_data) + ''');

                chart.set("zoomControl", am5map.ZoomControl.new(root, {}));
                
                chart.appear(1000, 100);
            });
            </script>
        </body>
        </html>
        '''
        
        # Display the map
        components.html(map_html, height=800)

        # Add legend in an expander
        with st.expander("Show Cluster Legend"):
            cols = st.columns(5)  # Adjust number of columns as needed
            for idx, (cluster, color) in enumerate(legend_data.items()):
                col_idx = idx % 5  # Adjust based on number of columns
                with cols[col_idx]:
                    st.markdown(f'<div style="display: flex; align-items: center;">'
                              f'<div style="width: 20px; height: 20px; background-color: {color}; margin-right: 8px;"></div>'
                              f'<span>{cluster}</span></div>', 
                              unsafe_allow_html=True)
        st.success('Map loaded successfully!')

def combine_sheet_data(df_dict, selected_sheets, numerics):
    combined_df = pd.DataFrame()
    
    for sheet in selected_sheets:
        sheet_df = df_dict[sheet]
        # Keep province info from first two columns for the first sheet
        if combined_df.empty:
            combined_df = sheet_df.iloc[:, :2].copy()
        
        # Add numeric columns, summing if they already exist
        numeric_cols = sheet_df.select_dtypes(include=numerics).columns
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] += sheet_df[col]
            else:
                combined_df[col] = sheet_df[col]
    
    return combined_df

def display_cluster_bar_charts(df, years):
    st.header("Cluster Analysis by Year")
    st.write("""
    The charts below show the distribution of emissions across provinces for each year, 
    color-coded by their cluster assignments. This helps visualize how provinces in different 
    clusters compare in terms of their emission levels over time.
    """)
    # Get the number of unique clusters
    n_clusters = df['Cluster'].nunique()
    
    # Create colors for the number of clusters we have
    # Using the same colors as in the map visualization
    cluster_colors = {
        1: '#FF9999',  # Light red
        2: '#99FF99',  # Light green
        3: '#9999FF',  # Light blue
        4: '#FFFF99',  # Light yellow
        5: '#FF99FF',  # Light purple
        6: '#99FFFF',  # Light cyan
        7: '#FFB366',  # Light orange
        8: '#B366FF',  # Light violet
        9: '#66FFB3',  # Light mint
        10: '#FF66B3'  # Light pink
    }
    
    # Create color list based on number of clusters
    colors = [cluster_colors[i] for i in range(1, n_clusters + 1)]
    cmap = ListedColormap(colors)

    # Split years into groups of 8
    years_per_group = 8
    n_groups = math.ceil(len(years) / years_per_group)

    # A4 size in inches (11.69 x 8.27)
    a4_height_inch = 11.69
    a4_width_inch = 8.27
    margin_inch = 3 / 2.54

    # Calculate available space for the plot
    plot_width_inch = a4_width_inch - 2 * margin_inch
    plot_height_inch = a4_height_inch - 2 * margin_inch

    for group in range(n_groups):
        start_year = group * years_per_group
        end_year = min((group + 1) * years_per_group, len(years))
        group_years = years[start_year:end_year]

        n_years = len(group_years)
        n_cols = 2  # Number of columns in the subplot grid
        n_rows = math.ceil(n_years / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(plot_width_inch, plot_height_inch), dpi=300)
        axes = axes.flatten()

        for i, year in enumerate(group_years):
            # Sort the data by cluster and then by emission value
            sorted_data = df.sort_values(['Cluster', year])

            # Create the bar chart
            bars = axes[i].bar(range(len(sorted_data)), sorted_data[year], color=cmap(sorted_data['Cluster'] - 1))

            # Customize the plot
            axes[i].set_xlabel('Province', fontsize=6)
            axes[i].set_ylabel('Emission Amount', fontsize=6)
            axes[i].set_title(f'{year}', fontsize=6)
            axes[i].set_xticks(range(len(sorted_data)))
            axes[i].set_xticklabels(sorted_data['PROVINCE'], 
                rotation=90, 
                ha='center',  # Changed from 'right' to 'center'
                va='top',
                fontsize=4)
            axes[i].tick_params(axis='y', labelsize=4)
            axes[i].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            plt.subplots_adjust(bottom=0.2)

        # Remove unused subplots
        for i in range(n_years, len(axes)):
            fig.delaxes(axes[i])

        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=cmap(i)) for i in range(n_clusters)]
        fig.legend(handles, [f'Cluster {i+1}' for i in range(n_clusters)],
                  loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=min(n_clusters, 5), fontsize=5)

        # Increase bottom margin and adjust spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.15, hspace=1, wspace=0.3)
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close()

if uploaded_file is not None:
    try:
        # Check the filename and its extension
        if isinstance(uploaded_file, str):  # If it's the example file
            if uploaded_file.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, sheet_name=None)
        else:  # If it's an uploaded file
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, sheet_name=None)

        # Get list of sheet names
        sheet_names = list(df.keys())
        
        # Move sheet selection to sidebar with "Select All" checkbox
        select_all_sheets = st.sidebar.checkbox("Select all sectors", False)
        if select_all_sheets:
            selected_sheets = sheet_names
        else:
            selected_sheets = st.sidebar.multiselect("Select sector(s) to preview", sheet_names)

        if selected_sheets:
            for sheet in selected_sheets:
                st.subheader(f"Preview of {sheet}")
                
                # Wrap data preview in an expander
                with st.expander("Data Preview"):
                    st.markdown("### Data preview")
                    display_data(df[sheet])
                
                # Apply preprocessing
                df[sheet] = preprocess_data(df[sheet])
                
                # Wrap preprocessed data in an expander
                with st.expander("Preprocessed Data"):
                    st.markdown("### Data preprocessed")
                    display_data(df[sheet])

            # Combine data if more than one sheet is selected
            if len(selected_sheets) > 1:
                combined_df = combine_sheet_data(df, selected_sheets, numerics)
                st.subheader(f"Preview of All Selected Sectors")
                with st.expander("Data Preview and Preprocessed"):
                    st.markdown("### Data preprocessed and combined")
                    display_data(combined_df)
            
            st.success("Data loaded and preprocessed successfully!")
        else:
            st.info("Please select at least one sector to preview from side bar.")
        
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a file to proceed.")

# Proceed with clustering only if df is not None and selected_sheets is not empty
if df is not None and selected_sheets:
    
    st.header("K-Means Clustering")

    if len(selected_sheets) > 1:
        # Add "All sectors" to the tabs options
        tab_options = selected_sheets + ["ALL SELECTED SECTORS"]
    else:
        tab_options = selected_sheets

    df_selected_sheet_option = ui.tabs(options=tab_options, default_value=selected_sheets[0])

    if df_selected_sheet_option == "ALL SELECTED SECTORS":
        # Use combined dataframe
        working_df = combined_df
    else:
        # Use selected sheet
        working_df = df[df_selected_sheet_option]

    # Get numeric columns for clustering
    num_cols = list(working_df.select_dtypes(include=numerics).columns)

    # Move year selection checkbox and multiselect to sidebar
    include_all_year = st.sidebar.checkbox("Select all years for clustering", value=False)

    if include_all_year:
        columns_for_model = sorted(num_cols)
    else:
        columns_for_model = st.sidebar.multiselect("Select at least two years for clustering", sorted(num_cols))

    if len(columns_for_model) == 1:
        st.write("Please choose at least one more year")
    elif len(columns_for_model) >= 2:
        perform_clustering(working_df, columns_for_model)
        # Only show bar charts after clustering is done and Cluster column exists
        if 'Cluster' in working_df.columns:
            display_cluster_bar_charts(working_df, columns_for_model)
        else:
            st.warning("Please perform clustering first before viewing the bar charts")



















