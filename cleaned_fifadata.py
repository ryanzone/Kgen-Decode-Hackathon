import pandas as pd

print("Loading datasets...")

# Load the datasets
matches_df = pd.read_csv("matches.csv")
players_21_df = pd.read_csv("players_21.csv")
players_22_df = pd.read_csv("players_22.csv")

print("Initial checks:")
print(matches_df.info())
print(players_21_df.info())
print(players_22_df.info())

print("\nCleaning matches.csv...")

# Clean matches.csv
matches_cleaned = matches_df.copy()

# Drop unnecessary columns
matches_cleaned.drop(columns=["Unnamed: 0", "notes"], inplace=True, errors='ignore')

# Convert 'date' and 'time' to datetime
matches_cleaned["date"] = pd.to_datetime(matches_cleaned["date"], errors='coerce')
matches_cleaned["time"] = pd.to_datetime(matches_cleaned["time"], format='%H:%M', errors='coerce').dt.time

# Fill missing attendance with median
matches_cleaned["attendance"].fillna(matches_cleaned["attendance"].median(), inplace=True)

# Fill missing distance with mean
matches_cleaned["dist"].fillna(matches_cleaned["dist"].mean(), inplace=True)

print("matches.csv cleaned ✅")

print("\nCleaning players_21.csv and players_22.csv...")

# Cleaning function
def clean_players(df):
    df_cleaned = df.copy()

    # Drop URL/logo-related columns and extras
    cols_to_drop = [col for col in df_cleaned.columns if "url" in col.lower()]
    extra_cols = ['player_face_url', 'club_logo_url', 'nation_logo_url', 'team_jersey_number']
    cols_to_drop += [col for col in extra_cols if col in df_cleaned.columns]
    df_cleaned.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Remove duplicates
    df_cleaned.drop_duplicates(inplace=True)

    # Clean position rating columns (like "52+3") into numbers
    for col in df_cleaned.select_dtypes(include="object"):
        if df_cleaned[col].astype(str).str.contains(r"\+\d+", na=False).any():
            df_cleaned[col] = df_cleaned[col].str.extract(r"(\d+)").astype(float)

    return df_cleaned

# Clean both player files
players_21_cleaned = clean_players(players_21_df)
players_22_cleaned = clean_players(players_22_df)

print("players_21.csv and players_22.csv cleaned ✅")

print("\nSaving cleaned files...")

# Save cleaned CSVs
matches_cleaned.to_csv("matches_cleaned.csv", index=False)
players_21_cleaned.to_csv("players_21_cleaned.csv", index=False)
players_22_cleaned.to_csv("players_22_cleaned.csv", index=False)

print("All cleaned files saved successfully ✅")
