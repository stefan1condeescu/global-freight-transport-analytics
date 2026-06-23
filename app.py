import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


DATA_FILE = "processed_transport_data.csv"

AIR_FREIGHT = "Air transport, freight (million ton-km)"
RAIL_FREIGHT = "Railways, goods transported (million ton-km)"
GDP = "GDP (current US$)"
POPULATION = "Population, total"

NUMERIC_ANALYSIS_COLUMNS = [AIR_FREIGHT, RAIL_FREIGHT, GDP, POPULATION]
MISSING_VALUE_COLUMNS = [AIR_FREIGHT, RAIL_FREIGHT]


st.set_page_config(page_title="Global Freight Transport Analytics", layout="wide")
st.title("Global Freight Transport Analytics")
st.caption(
    "Interactive Streamlit dashboard for exploring air freight, rail freight, GDP, "
    "population, regions, and income groups."
)


@st.cache_data
def load_transport_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def build_scaler(method: str):
    if method == "StandardScaler":
        return StandardScaler()
    return MinMaxScaler()


df = load_transport_data()

st.header("1. Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)


st.header("2. Missing Value Analysis")
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

if missing_values.empty:
    st.success("No missing values were found in the dataset.")
else:
    st.warning(f"{len(missing_values)} columns contain missing values.")
    st.dataframe(
        pd.DataFrame(
            {
                "Column": missing_values.index,
                "Missing values": missing_values.values,
                "Missing percentage": (missing_values.values / len(df) * 100).round(2),
            }
        ),
        use_container_width=True,
    )

st.subheader("Rows With Missing Values")
if missing_values.empty:
    st.info("There are no missing-value columns to inspect.")
else:
    selected_missing_column = st.selectbox(
        "Choose a column to inspect:",
        list(missing_values.index),
    )
    st.write("Rows where the selected column is missing:")
    st.dataframe(df[df[selected_missing_column].isnull()], use_container_width=True)

st.header("Missing Value Treatment")
st.subheader("Available Treatment Methods by Variable Type")

variable_type = st.radio(
    "Choose the variable type:",
    ["Numerical", "Categorical"],
)

if variable_type == "Numerical":
    st.write("For numerical variables, common treatment methods include:")
    st.markdown(
        """
        - **Mean** - useful when values are distributed relatively evenly.
        - **Median** - useful when extreme values are present.
        - **Forward fill** - fills each missing value with the previous value.
        - **Backward fill** - fills each missing value with the next value.
        - **Row removal** - useful only when very few rows are affected.
        """
    )
else:
    st.write("For categorical variables, common treatment methods include:")
    st.markdown(
        """
        - **Mode** - fills missing values with the most frequent category.
        - **Unknown** - useful when no category should be assumed.
        - **Row removal** - useful only when very few rows are affected.
        """
    )

cleaned_df = df.copy()
cleaned_df = cleaned_df.sort_values(by=["Country Name", "Year"])

for column in MISSING_VALUE_COLUMNS:
    cleaned_df[column] = cleaned_df.groupby("Country Name")[column].ffill()
    cleaned_df[column] = cleaned_df.groupby("Country Name")[column].bfill()

for column in MISSING_VALUE_COLUMNS:
    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

st.subheader("Missing Values After Treatment")
missing_after_treatment = cleaned_df.isnull().sum()
missing_after_treatment = missing_after_treatment[missing_after_treatment > 0]

if missing_after_treatment.empty:
    st.success("All missing values were treated successfully.")
else:
    st.warning("The following columns still contain missing values:")
    st.dataframe(
        pd.DataFrame(
            {
                "Column": missing_after_treatment.index,
                "Missing values": missing_after_treatment.values,
                "Missing percentage": (
                    missing_after_treatment.values / len(cleaned_df) * 100
                ).round(2),
            }
        ),
        use_container_width=True,
    )

with st.expander("Show cleaned dataset"):
    st.dataframe(cleaned_df, use_container_width=True)

cleaned_csv = cleaned_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download cleaned dataset",
    data=cleaned_csv,
    file_name="cleaned_transport_data.csv",
    mime="text/csv",
)


st.header("3. Outlier Detection")
available_outlier_columns = [
    column for column in NUMERIC_ANALYSIS_COLUMNS if column in cleaned_df.columns
]

st.subheader("IQR-Based Outlier Analysis")
selected_outlier_column = st.selectbox(
    "Choose a column for outlier analysis:",
    available_outlier_columns,
)

q1 = cleaned_df[selected_outlier_column].quantile(0.25)
q3 = cleaned_df[selected_outlier_column].quantile(0.75)
iqr = q3 - q1

lower_limit = max(0, q1 - 1.5 * iqr)
upper_limit = q3 + 1.5 * iqr

outlier_rows = cleaned_df[
    (cleaned_df[selected_outlier_column] < lower_limit)
    | (cleaned_df[selected_outlier_column] > upper_limit)
]

outlier_count = len(outlier_rows)
outlier_percentage = round(outlier_count / len(cleaned_df) * 100, 2)

st.subheader("Outlier Detection Results")
metric_col_1, metric_col_2 = st.columns(2)

with metric_col_1:
    st.metric("Lower limit", f"{lower_limit:,.0f}")
    st.metric("Outlier count", outlier_count)

with metric_col_2:
    st.metric("Upper limit", f"{upper_limit:,.0f}")
    st.metric("Outlier percentage", f"{outlier_percentage:.2f}%")

with st.expander("Show rows identified as outliers"):
    st.dataframe(outlier_rows, use_container_width=True)

st.subheader("Box Plot and Histogram")
chart_col_1, chart_col_2 = st.columns(2)

with chart_col_1:
    box_plot = px.box(
        cleaned_df,
        y=selected_outlier_column,
        title=f"Box plot for {selected_outlier_column} (log scale)",
    )
    box_plot.update_layout(height=330, margin=dict(l=10, r=10, t=55, b=10))
    box_plot.update_yaxes(type="log")
    st.plotly_chart(box_plot, use_container_width=True)

with chart_col_2:
    histogram = px.histogram(
        cleaned_df,
        x=selected_outlier_column,
        nbins=30,
        title=f"Histogram for {selected_outlier_column}",
    )
    histogram.update_layout(height=330, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(histogram, use_container_width=True)


st.header("4. Categorical Variable Encoding")
categorical_columns = cleaned_df.select_dtypes(include=["object"]).columns.tolist()
excluded_categorical_columns = ["Country Name"]
categorical_columns = [
    column for column in categorical_columns if column not in excluded_categorical_columns
]

st.subheader("Available Categorical Columns")
if categorical_columns:
    st.write(categorical_columns)
else:
    st.info("No categorical columns are available for encoding.")

selected_encoding_columns = st.multiselect(
    "Choose columns to encode:",
    categorical_columns,
    default=categorical_columns,
)

encoding_method = st.selectbox(
    "Choose an encoding method:",
    ["One-Hot Encoding", "Label Encoding"],
)

if selected_encoding_columns:
    encoded_df = cleaned_df.copy()

    if encoding_method == "One-Hot Encoding":
        encoded_df = pd.get_dummies(
            encoded_df,
            columns=selected_encoding_columns,
            drop_first=False,
        )
    else:
        label_encoder = LabelEncoder()
        for column in selected_encoding_columns:
            encoded_df[column] = label_encoder.fit_transform(
                encoded_df[column].astype(str)
            )

    st.subheader("Encoding Result")
    metric_col_1, metric_col_2 = st.columns(2)

    with metric_col_1:
        st.metric("Original column count", cleaned_df.shape[1])

    with metric_col_2:
        st.metric("Encoded column count", encoded_df.shape[1])

    st.write("First rows:")
    st.dataframe(encoded_df.head(20), use_container_width=True)

    encoded_csv = encoded_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download encoded dataset",
        data=encoded_csv,
        file_name="encoded_transport_data.csv",
        mime="text/csv",
    )
else:
    st.warning("Select at least one column to encode.")


st.header("5. Feature Scaling")
available_scaling_columns = [
    column for column in NUMERIC_ANALYSIS_COLUMNS if column in cleaned_df.columns
]

st.subheader("Available Numerical Columns")
st.write(", ".join(available_scaling_columns))

selected_scaling_columns = st.multiselect(
    "Choose columns to scale:",
    available_scaling_columns,
    default=available_scaling_columns,
)

scaling_method = st.selectbox(
    "Choose a scaling method:",
    ["StandardScaler", "MinMaxScaler"],
)

scaled_df = cleaned_df.copy()
if selected_scaling_columns:
    display_scaler = build_scaler(scaling_method)
    scaled_df[selected_scaling_columns] = display_scaler.fit_transform(
        scaled_df[selected_scaling_columns]
    )

    st.subheader("Scaling Results")
    before_col, after_col = st.columns(2)

    with before_col:
        st.write("Before scaling:")
        st.dataframe(cleaned_df[selected_scaling_columns].head(10), use_container_width=True)

    with after_col:
        st.write("After scaling:")
        st.dataframe(scaled_df[selected_scaling_columns].head(10), use_container_width=True)
else:
    st.warning("Select at least one column to scale.")

model_scaled_df = cleaned_df.copy()
model_scaler = build_scaler(scaling_method)
model_scaled_df[available_scaling_columns] = model_scaler.fit_transform(
    model_scaled_df[available_scaling_columns]
)


st.header("6. Descriptive Statistics, Grouping, and Aggregation")
st.subheader("A. General Descriptive Statistics")
available_numeric_columns = [
    column for column in NUMERIC_ANALYSIS_COLUMNS if column in cleaned_df.columns
]

if available_numeric_columns:
    descriptive_stats = cleaned_df[available_numeric_columns].describe().T.round(2)
    st.dataframe(descriptive_stats, use_container_width=True)
else:
    st.info("No numerical columns are available for analysis.")

st.subheader("B. Freight Transport Evolution Over Time")
if "Year" in cleaned_df.columns:
    yearly_aggregation = cleaned_df.groupby("Year").agg(
        {
            AIR_FREIGHT: "mean",
            RAIL_FREIGHT: "mean",
            GDP: "mean",
            POPULATION: "mean",
        }
    ).round(2)

    st.write("Average yearly values for the main indicators:")
    st.dataframe(yearly_aggregation, use_container_width=True)

    st.line_chart(
        yearly_aggregation[
            [
                AIR_FREIGHT,
                RAIL_FREIGHT,
            ]
        ]
    )
else:
    st.info("The dataset does not contain a 'Year' column.")

st.subheader("C. Indicator Comparison by Geographic Region")
if "Region" in cleaned_df.columns:
    regional_aggregation = cleaned_df.groupby("Region").agg(
        {
            AIR_FREIGHT: "mean",
            RAIL_FREIGHT: "mean",
            GDP: "mean",
            POPULATION: "mean",
        }
    ).round(2)

    st.write("Average values of the main indicators by region:")
    st.dataframe(regional_aggregation, use_container_width=True)

    st.write("Air freight and rail freight comparison by region:")
    st.bar_chart(
        regional_aggregation[
            [
                AIR_FREIGHT,
                RAIL_FREIGHT,
            ]
        ]
    )
else:
    st.info("The dataset does not contain a 'Region' column.")


st.header("7. Filtering and Sorting by Air Freight")
filtered_df = cleaned_df.copy()

min_air_freight = float(filtered_df[AIR_FREIGHT].min())
max_air_freight = float(filtered_df[AIR_FREIGHT].max())

selected_air_freight_range = st.slider(
    "Select the air freight interval (million ton-km):",
    min_value=min_air_freight,
    max_value=max_air_freight,
    value=(1.0, max_air_freight),
)

filtered_df = filtered_df[
    (filtered_df[AIR_FREIGHT] >= selected_air_freight_range[0])
    & (filtered_df[AIR_FREIGHT] <= selected_air_freight_range[1])
]

sort_order = st.radio(
    "Choose the sorting order:",
    ["Descending", "Ascending"],
)

filtered_df = filtered_df.sort_values(
    by=AIR_FREIGHT,
    ascending=sort_order == "Ascending",
)

st.write(f"Observation count: {len(filtered_df)}")
st.dataframe(filtered_df.head(20), use_container_width=True)


st.header("8. Regional Market Share Analysis")
market_share_df = cleaned_df.copy()
market_share_df["Region_Year_Total"] = market_share_df.groupby(["Region", "Year"])[
    AIR_FREIGHT
].transform("sum")
market_share_df["Regional_Share"] = (
    market_share_df[AIR_FREIGHT] / market_share_df["Region_Year_Total"] * 100
).round(2)

selected_year = st.selectbox(
    "Select the year for market share analysis:",
    sorted(market_share_df["Year"].unique(), reverse=True),
)

year_market_share_df = market_share_df[market_share_df["Year"] == selected_year].sort_values(
    by="Regional_Share",
    ascending=False,
)

st.write(
    f"Countries with the highest regional air freight share in {selected_year}:"
)
st.dataframe(
    year_market_share_df[
        ["Country Name", "Region", AIR_FREIGHT, "Regional_Share"]
    ].head(10),
    use_container_width=True,
)

market_share_chart = px.bar(
    year_market_share_df.head(15),
    x="Country Name",
    y="Regional_Share",
    color="Region",
    title="Country share inside its geographic region (%)",
)
st.plotly_chart(market_share_chart, use_container_width=True)


st.header("9. Country Clustering With K-Means")
clustering_columns = [AIR_FREIGHT, RAIL_FREIGHT, GDP, POPULATION]

kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=10)
clusters = kmeans.fit_predict(model_scaled_df[clustering_columns])

clustered_df = cleaned_df.copy()
clustered_df["Cluster"] = clusters
clustered_df["Cluster_Label"] = clustered_df["Cluster"].map(
    lambda cluster_id: f"Cluster {cluster_id}"
)

cluster_chart_df = clustered_df[
    (clustered_df[GDP] > 0) & (clustered_df[AIR_FREIGHT] > 0)
].copy()

cluster_colors = {
    "Cluster 0": "#2563EB",
    "Cluster 1": "#F97316",
    "Cluster 2": "#10B981",
}

cluster_scatter = px.scatter(
    cluster_chart_df,
    x=GDP,
    y=AIR_FREIGHT,
    color="Cluster_Label",
    color_discrete_map=cluster_colors,
    hover_name="Country Name",
    log_x=True,
    log_y=True,
    title="Country clusters: GDP and air freight relationship",
    labels={"Cluster_Label": "Cluster"},
)
cluster_scatter.update_traces(marker=dict(size=8, opacity=0.82, line=dict(width=0.7)))
st.plotly_chart(cluster_scatter, use_container_width=True)

st.subheader("Explore Countries by Cluster")
selected_cluster = st.selectbox(
    "Choose a cluster:",
    sorted(clustered_df["Cluster"].unique()),
)

cluster_details_df = clustered_df[clustered_df["Cluster"] == selected_cluster]

st.write(f"Cluster {selected_cluster} contains {len(cluster_details_df)} observations:")
st.dataframe(
    cluster_details_df[
        ["Country Name", "Region", GDP, AIR_FREIGHT]
    ],
    use_container_width=True,
)


st.header("10. Multiple Linear Regression")
regression_features = [GDP, POPULATION, RAIL_FREIGHT]
X_regression = model_scaled_df[regression_features]
y_regression = model_scaled_df[AIR_FREIGHT]

linear_model = LinearRegression()
linear_model.fit(X_regression, y_regression)

y_regression_pred = linear_model.predict(X_regression)
r2 = r2_score(y_regression, y_regression_pred)

st.write(f"**Coefficient of determination (R^2):** {r2:.4f}")

st.subheader("Estimated Influence of Each Factor")
feature_importance = pd.DataFrame(
    {
        "Factor": ["GDP", "Population", "Rail freight"],
        "Coefficient": linear_model.coef_,
    }
).sort_values(by="Coefficient", ascending=False)

st.table(feature_importance)

st.info(
    """
    A positive coefficient means the model associates higher values of that factor
    with higher air freight values. R^2 shows how much of the air freight variation
    is explained by the selected factors in this in-sample educational model.
    """
)


st.header("11. Logistic Regression: Hub Status Prediction")
hub_threshold = model_scaled_df[AIR_FREIGHT].quantile(0.75)
model_scaled_df["Is_Hub"] = (model_scaled_df[AIR_FREIGHT] > hub_threshold).astype(int)

X_logistic = model_scaled_df[[GDP, POPULATION]]
y_logistic = model_scaled_df["Is_Hub"]

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_logistic, y_logistic)

y_logistic_pred = logistic_model.predict(X_logistic)
accuracy = accuracy_score(y_logistic, y_logistic_pred)

st.write(f"**In-sample model accuracy:** {accuracy:.4f}")
st.write(
    "The model classifies whether a country-year observation belongs to the top "
    f"25% of air freight values with {accuracy * 100:.2f}% in-sample accuracy."
)

st.write("**Confusion Matrix**")
confusion = confusion_matrix(y_logistic, y_logistic_pred)

confusion_chart = px.imshow(
    confusion,
    text_auto=True,
    aspect="auto",
    labels={"x": "Model prediction", "y": "Actual class"},
    x=["Non-Hub", "Hub"],
    y=["Non-Hub", "Hub"],
    color_continuous_scale="Blues",
    title="Prediction distribution: Hub vs Non-Hub",
)

confusion_chart.update_layout(
    xaxis_title="Model prediction",
    yaxis_title="Actual class",
)

st.plotly_chart(confusion_chart, use_container_width=True)

with st.expander("Show detailed classification report"):
    report_dict = classification_report(
        y_logistic,
        y_logistic_pred,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    st.write("Performance statistics: precision, recall, and F1-score")
    st.table(report_df.round(2))

    st.info(
        """
        Precision shows how reliable the model is when it predicts a hub.
        Recall shows how many true hubs the model identifies.
        F1-score combines precision and recall into a single score.
        """
    )
