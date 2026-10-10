
import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Universal Data Analytics",
    page_icon="📊",
    layout="wide"
)

MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_ROWS_WARNING = 200_000
MAX_CATEGORIES = 20

st.title("📊 Universal Data Analytics Dashboard")
st.write(
    "Upload any supported dataset to explore, clean, "
    "filter and visualize your data."
)


# ==========================================
# 2. LOAD AND VALIDATE DATA
# ==========================================

def load_dataset(uploaded_file):
    raw = uploaded_file.getvalue()

    if not raw:
        raise ValueError("The uploaded file is empty.")

    if len(raw) > MAX_FILE_SIZE:
        raise ValueError("Maximum file size is 50 MB.")

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        data = pd.read_csv(io.BytesIO(raw))
    elif filename.endswith(".xlsx"):
        data = pd.read_excel(
            io.BytesIO(raw),
            engine="openpyxl"
        )
    else:
        raise ValueError("Please upload a CSV or XLSX file.")

    if len(data.columns) == 0:
        raise ValueError("No columns were found.")

    if data.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names found. "
            "Please rename them before uploading."
        )

    return data


# ==========================================
# 3. DATE DETECTION
# ==========================================

def detect_date_columns(data):
    date_columns = []

    for column in data.columns:
        series = data[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(column)
            continue

        if not (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            continue

        sample = series.dropna().head(2000)

        if sample.empty:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(
                    sample,
                    errors="coerce",
                    format="mixed"
                )
        except (ValueError, TypeError):
            parsed = pd.to_datetime(
                sample,
                errors="coerce"
            )

        if parsed.notna().mean() >= 0.85:
            date_columns.append(column)

    return date_columns


# ==========================================
# 4. UPLOAD SECTION
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to begin.")
    st.stop()

try:
    original_df = load_dataset(uploaded_file)
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

if original_df.empty:
    st.warning("Your dataset contains no records.")
    st.stop()

# Never modify the original DataFrame.
original_df = original_df.copy(deep=True)

if len(original_df) > MAX_ROWS_WARNING:
    st.warning(
        "Large dataset detected. "
        "Processing may take additional time and memory."
    )


# ==========================================
# 5. DATA CLEANING CONTROLS
# ==========================================

st.sidebar.header("Data Cleaning")

remove_duplicates = st.sidebar.checkbox(
    "Remove duplicate rows"
)

convert_dates = st.sidebar.checkbox(
    "Convert detected date columns"
)

missing_method = st.sidebar.selectbox(
    "Missing-value treatment",
    [
        "Keep missing values",
        "Drop rows with missing values",
        "Fill numeric with mean",
        "Fill numeric with median",
        "Fill numeric with zero",
        "Fill with mode / Unknown"
    ]
)

# Create a fresh working copy on every rerun.
df = original_df.copy(deep=True)

if remove_duplicates:
    df = df.drop_duplicates().copy()

detected_dates = detect_date_columns(df)

if convert_dates:
    for column in detected_dates:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

if missing_method == "Drop rows with missing values":
    df = df.dropna().copy()

elif missing_method in [
    "Fill numeric with mean",
    "Fill numeric with median",
    "Fill numeric with zero"
]:
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:
        if missing_method == "Fill numeric with mean":
            fill_value = df[column].mean()
        elif missing_method == "Fill numeric with median":
            fill_value = df[column].median()
        else:
            fill_value = 0

        if pd.notna(fill_value):
            df[column] = df[column].fillna(fill_value)

elif missing_method == "Fill with mode / Unknown":
    for column in df.columns:
        if not df[column].isna().any():
            continue

        modes = df[column].mode(dropna=True)

        if not modes.empty:
            df[column] = df[column].fillna(modes.iloc[0])

        elif (
            pd.api.types.is_object_dtype(df[column])
            or isinstance(df[column].dtype, pd.CategoricalDtype)
        ):
            df[column] = (
                df[column]
                .astype("object")
                .fillna("Unknown")
            )

date_columns = detect_date_columns(df)


# ==========================================
# 6. INTERACTIVE FILTERS
# ==========================================

st.sidebar.header("Dashboard Filters")

filtered_df = df.copy()

for column in df.columns:
    series = df[column]

    # DATE FILTER
    if column in date_columns:
        dates = pd.to_datetime(
            series,
            errors="coerce"
        )

        valid_dates = dates.dropna()

        if valid_dates.empty:
            continue

        date_range = st.sidebar.date_input(
            f"Date range: {column}",
            value=(
                valid_dates.min().date(),
                valid_dates.max().date()
            ),
            key=f"date_{column}"
        )

        if isinstance(date_range, (tuple, list)):
            if len(date_range) == 2:
                start_date, end_date = date_range

                current_dates = pd.to_datetime(
                    filtered_df[column],
                    errors="coerce"
                )

                filtered_df = filtered_df[
                    current_dates.dt.date.between(
                        start_date,
                        end_date
                    )
                ]

    # NUMERIC FILTER
    elif pd.api.types.is_numeric_dtype(series):
        values = series.dropna()

        if (
            values.empty
            or values.nunique() <= 1
            or values.nunique() > 10_000
        ):
            continue

        lower = float(values.min())
        upper = float(values.max())

        selected_range = st.sidebar.slider(
            f"Range: {column}",
            min_value=lower,
            max_value=upper,
            value=(lower, upper),
            key=f"numeric_{column}"
        )

        filtered_df = filtered_df[
            filtered_df[column].between(
                selected_range[0],
                selected_range[1]
            )
        ]

    # CATEGORICAL FILTER
    elif series.nunique(dropna=True) <= 50:
        options = sorted(
            series.dropna().astype(str).unique().tolist()
        )

        if options:
            selected_categories = st.sidebar.multiselect(
                f"Category: {column}",
                options=options,
                default=options,
                key=f"category_{column}"
            )

            filtered_df = filtered_df[
                filtered_df[column].isna()
                | filtered_df[column]
                    .astype(str)
                    .isin(selected_categories)
            ]


# ==========================================
# 7. DATASET OVERVIEW
# ==========================================

st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Original Rows",
    f"{len(original_df):,}"
)

col2.metric(
    "Working Rows",
    f"{len(df):,}"
)

col3.metric(
    "Filtered Rows",
    f"{len(filtered_df):,}"
)

col4.metric(
    "Missing Cells",
    f"{int(filtered_df.isna().sum().sum()):,}"
)

st.subheader("Data Preview")

st.dataframe(
    filtered_df.head(500),
    use_container_width=True
)


# ==========================================
# 8. DATA QUALITY REPORT
# ==========================================

st.header("Data Quality Report")

quality_report = pd.DataFrame({
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isna().sum(),
    "Missing Percentage": (
        df.isna().mean() * 100
    ).round(2),
    "Unique Values": df.nunique()
})

st.dataframe(
    quality_report,
    use_container_width=True
)


# ==========================================
# 9. STOP IF FILTER RESULTS ARE EMPTY
# ==========================================

if filtered_df.empty:
    st.warning("No rows match your current filters.")
    st.stop()


# ==========================================
# 10. SMART CHART RECOMMENDATIONS
# ==========================================

st.header("Smart Chart Recommendations")

numeric_columns = filtered_df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = [
    column
    for column in filtered_df.columns
    if (
        column not in date_columns
        and filtered_df[column].nunique(dropna=True) <= 50
        and (
            pd.api.types.is_object_dtype(filtered_df[column])
            or isinstance(
                filtered_df[column].dtype,
                pd.CategoricalDtype
            )
            or pd.api.types.is_bool_dtype(filtered_df[column])
        )
    )
]

# NUMERIC DISTRIBUTION
if numeric_columns:
    numeric_column = numeric_columns[0]

    fig = px.histogram(
        filtered_df,
        x=numeric_column,
        marginal="box",
        title=f"Distribution of {numeric_column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# CATEGORY COMPARISON
if categorical_columns:
    category_column = categorical_columns[0]

    category_counts = (
        filtered_df[category_column]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .head(MAX_CATEGORIES)
        .rename_axis("Category")
        .reset_index(name="Count")
    )

    fig = px.bar(
        category_counts,
        x="Category",
        y="Count",
        title=f"Top Categories: {category_column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# NUMERIC RELATIONSHIP
if len(numeric_columns) >= 2:
    fig = px.scatter(
        filtered_df,
        x=numeric_columns[0],
        y=numeric_columns[1],
        title="Relationship Between Numerical Columns"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# CORRELATION
if len(numeric_columns) >= 2:
    st.subheader("Correlation Analysis")

    correlation = filtered_df[
        numeric_columns
    ].corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# 11. DATE INTELLIGENCE
# ==========================================

if date_columns and numeric_columns:
    st.header("Date Intelligence")

    date_column = st.selectbox(
        "Select date column",
        date_columns
    )

    metric_column = st.selectbox(
        "Select numerical metric",
        numeric_columns
    )

    frequency = st.selectbox(
        "Time grouping",
        [
            "Daily",
            "Weekly",
            "Monthly",
            "Quarterly",
            "Yearly"
        ]
    )

    frequency_map = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "ME",
        "Quarterly": "QE",
        "Yearly": "YE"
    }

    time_df = filtered_df[
        [date_column, metric_column]
    ].copy()

    time_df[date_column] = pd.to_datetime(
        time_df[date_column],
        errors="coerce"
    )

    time_df = time_df.dropna()

    if not time_df.empty:
        trend = (
            time_df
            .set_index(date_column)[metric_column]
            .resample(frequency_map[frequency])
            .sum()
            .reset_index()
        )

        fig = px.line(
            trend,
            x=date_column,
            y=metric_column,
            markers=True,
            title=f"{frequency} Total: {metric_column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==========================================
# 12. CUSTOM CHART BUILDER
# ==========================================

st.header("Custom Chart Builder")

x_column = st.selectbox(
    "Select X-axis",
    filtered_df.columns.tolist()
)

chart_type = st.selectbox(
    "Select chart type",
    ["Bar", "Line", "Scatter", "Histogram", "Box"]
)

y_column = None

if chart_type != "Histogram" and numeric_columns:
    y_column = st.selectbox(
        "Select Y-axis",
        numeric_columns
    )

fig = None

if chart_type == "Histogram":
    fig = px.histogram(
        filtered_df,
        x=x_column
    )

elif y_column is not None:
    if chart_type == "Bar":
        chart_data = (
            filtered_df
            .groupby(x_column, dropna=False)[y_column]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            chart_data,
            x=x_column,
            y=y_column
        )

    elif chart_type == "Line":
        fig = px.line(
            filtered_df,
            x=x_column,
            y=y_column
        )

    elif chart_type == "Scatter":
        fig = px.scatter(
            filtered_df,
            x=x_column,
            y=y_column
        )

    elif chart_type == "Box":
        fig = px.box(
            filtered_df,
            x=x_column,
            y=y_column
        )

if fig is not None:
    st.plotly_chart(
        fig,
        use_container_width=True
    )
else:
    st.info("Select a numerical Y-axis for this chart.")


# ==========================================
# 13. DOWNLOAD RESULTS
# ==========================================

st.header("Download Results")

col1, col2 = st.columns(2)

col1.download_button(
    label="Download Cleaned Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_dataset.csv",
    mime="text/csv"
)

col2.download_button(
    label="Download Filtered Dataset",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_dataset.csv",
    mime="text/csv"
)

st.success("Your data analysis is complete!")

st.caption(
    "Cleaning operations use a working copy. "
    "The original uploaded DataFrame remains unchanged."
)
