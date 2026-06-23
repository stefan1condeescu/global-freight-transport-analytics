import numpy as np
import pandas as pd


RAW_DATA_FILE = "raw_transport_data.csv"
COUNTRY_METADATA_FILE = "country_metadata.csv"
PROCESSED_DATA_FILE = "processed_transport_data.csv"


def build_processed_dataset() -> pd.DataFrame:
    raw_df = pd.read_csv(RAW_DATA_FILE)

    # Keep the original World Bank export unchanged and build a working dataset.
    working_df = raw_df.copy()

    # Remove columns that are not needed in the dashboard.
    working_df = working_df.drop(columns=["Series Code"])

    # Remove trailing World Bank metadata rows that do not describe countries or series.
    working_df = working_df.dropna(subset=["Series Name", "Country Name"])

    # World Bank exports missing numeric values as "..".
    working_df = working_df.replace("..", np.nan)

    # Convert year columns from text to numeric values.
    year_columns = [column for column in working_df.columns if "[" in column]
    for column in year_columns:
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    # Convert the dataset from wide format to long format.
    working_df = working_df.melt(
        id_vars=["Country Name", "Series Name", "Country Code"],
        var_name="Year",
        value_name="Value",
    )

    # Extract the numeric year from labels such as "2015 [YR2015]".
    working_df["Year"] = working_df["Year"].str.split(" ").str[0].astype(int)

    # Convert indicator names from rows into individual columns.
    working_df = working_df.pivot_table(
        index=["Country Name", "Year", "Country Code"],
        columns="Series Name",
        values="Value",
    ).reset_index()

    # Add geographic region and income group metadata.
    metadata_df = pd.read_csv(COUNTRY_METADATA_FILE)
    metadata_df = metadata_df[["Code", "Region", "Income Group"]]

    working_df = pd.merge(
        working_df,
        metadata_df,
        left_on="Country Code",
        right_on="Code",
        how="left",
    )

    return working_df.drop(columns=["Code", "Country Code"])


if __name__ == "__main__":
    processed_df = build_processed_dataset()
    processed_df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(
        f"Saved {PROCESSED_DATA_FILE} with "
        f"{processed_df.shape[0]} rows and {processed_df.shape[1]} columns."
    )

