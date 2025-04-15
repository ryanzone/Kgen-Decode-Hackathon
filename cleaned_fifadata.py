import pandas as pd

# Load the datasets
matches_df = pd.read_csv("matches.csv")
players_21_df = pd.read_csv("players_21.csv")
players_22_df = pd.read_csv("players_22.csv")

# Check basic info and preview the datasets
matches_info = matches_df.info()
players_21_info = players_21_df.info()
players_22_info = players_22_df.info()

# View head of each dataframe
matches_head = matches_df.head()
players_21_head = players_21_df.head()
players_22_head = players_22_df.head()





# Cleaning matches.csv
matches_cleaned = matches_df.copy()

# Drop unnecessary columns
matches_cleaned.drop(columns=["Unnamed: 0", "notes"], inplace=True)

# Convert 'date' and 'time' to datetime format
matches_cleaned["date"] = pd.to_datetime(matches_cleaned["date"])
matches_cleaned["time"] = pd.to_datetime(matches_cleaned["time"], format='%H:%M', errors='coerce').dt.time

# Fill missing 'attendance' with median
matches_cleaned["attendance"].fillna(matches_cleaned["attendance"].median(), inplace=True)

# Fill missing 'dist' with mean
matches_cleaned["dist"].fillna(matches_cleaned["dist"].mean(), inplace=True)


# Cleaning players_21.csv and players_22.csv
def clean_players(df):
    df_cleaned = df.copy()

    # Drop unnecessary URL/logo columns
    cols_to_drop = [col for col in df_cleaned.columns if "url" in col]
    df_cleaned.drop(columns=cols_to_drop, inplace=True)

    # Remove duplicates
    df_cleaned.drop_duplicates(inplace=True)

    # Clean position ratings (like "52+3") to numeric if possible
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == "object" and df_cleaned[col].str.contains(r"\+\d+", na=False).any():
            df_cleaned[col] = df_cleaned[col].str.extract(r"(\d+)").astype(float)

    return df_cleaned


players_21_cleaned = clean_players(players_21_df)
players_22_cleaned = clean_players(players_22_df)

# Save cleaned files
matches_cleaned.to_csv("matches_cleaned.csv", index=False)
players_21_cleaned.to_csv("players_21_cleaned.csv", index=False)
players_22_cleaned.to_csv("players_22_cleaned.csv", index=False)
