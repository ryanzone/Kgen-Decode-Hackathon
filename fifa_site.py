import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report

# --- Page config ---
st.set_page_config(page_title="FIFA AI Predictor", page_icon= "fifa.png", layout="wide")

st.title("FIFA AI Predictor")
st.markdown("Welcome to your interactive AI-based player and match forecasting tool!")

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")

# Correct mapping for cleaner labels but correct logic
page_options = {
    "Match Predictor": "Match Predictor",
    "Player Progress Predictor": "Player Progress Predictor from start of 2021 to end of 2022"
}
page_selection = st.sidebar.radio("Choose a feature", list(page_options.keys()))
page = page_options[page_selection]

# -------------------- MATCH PREDICTOR --------------------
if page == "Match Predictor":
    st.header("Match Outcome Predictor")
    st.markdown("Predict the result of a football match using match statistics and team history.")

    @st.cache_data
    def load_and_train_model():
        df = pd.read_csv("matches_cleaned.csv")

        # Label encoding
        label_cols = ["venue", "opponent", "formation", "day"]
        label_encoders = {}
        for col in label_cols:
            le = LabelEncoder()
            df[col + "_code"] = le.fit_transform(df[col])
            label_encoders[col] = le

        result_encoder = LabelEncoder()
        df["result_code"] = result_encoder.fit_transform(df["result"])

        features = ["venue_code", "opponent_code", "formation_code", "day_code",
                    "xg", "xga", "poss", "gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "attendance"]
        X = df[features]
        y = df["result_code"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        return model, label_encoders, scaler, result_encoder, df

    model, label_encoders, scaler, result_encoder, df = load_and_train_model()
    teams = df["team"].unique()
    opponents = df["opponent"].unique()

    with st.form("match_form"):
        st.subheader("Select Match Settings")
        col1, col2 = st.columns(2)
        with col1:
            team_a = st.selectbox("Home Team", teams)
        with col2:
            team_b = st.selectbox("Away Team", opponents)

        submitted = st.form_submit_button(" Predict Match Result")

    if submitted:
        if team_a == team_b:
            st.warning("Please choose two different teams to predict a match.")
        else:
            st.success(f"Predicting outcome for {team_a} vs {team_b}...")

            team_a_stats = df[df["team"] == team_a].mean(numeric_only=True)
            team_b_stats = df[df["opponent"] == team_b].mean(numeric_only=True)

            sample = df.iloc[0:1].copy()
            for col in ["xg", "xga", "poss", "gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "attendance"]:
                sample[col] = (team_a_stats[col] + team_b_stats[col]) / 2

            # Encode
            sample["venue_code"] = label_encoders["venue"].transform([sample["venue"].values[0]])[0]
            sample["opponent_code"] = label_encoders["opponent"].transform([team_b])[0]
            sample["formation_code"] = label_encoders["formation"].transform([sample["formation"].values[0]])[0]
            sample["day_code"] = label_encoders["day"].transform([sample["day"].values[0]])[0]

            feature_cols = ["venue_code", "opponent_code", "formation_code", "day_code",
                            "xg", "xga", "poss", "gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "attendance"]

            X_input = scaler.transform(sample[feature_cols])
            prediction = model.predict(X_input)[0]
            prediction_proba = model.predict_proba(X_input)[0]
            result_label = result_encoder.inverse_transform([prediction])[0]

            st.subheader("Predicted Outcome")
            st.markdown(f"**Match Result:** `{result_label}`")
            st.bar_chart(pd.DataFrame({
                "Result": result_encoder.inverse_transform([0, 1, 2]),
                "Probability": prediction_proba
            }).set_index("Result"))

    st.markdown("---")
    st.caption("Model based on historical match stats using a trained Random Forest Classifier.")

# -------------------- PLAYER PROGRESS --------------------
if page == "Player Progress Predictor from start of 2021 to end of 2022":
    st.title("Player Stats Comparison: 2021 vs 2022")

    @st.cache_data
    def load_data(year):
        return pd.read_csv(f"players_{year}_cleaned.csv", low_memory=False)

    df_21 = load_data("21")
    df_22 = load_data("22")

    st.subheader("Filter Options")

    nationality_filter = st.multiselect("Select Nationalities", 
        sorted(set(df_21["nationality_name"].unique()).union(df_22["nationality_name"].unique())))

    position_filter = st.multiselect("Select Positions", 
        sorted(set(df_21["player_positions"].unique()).union(df_22["player_positions"].unique())))

    if nationality_filter:
        df_21 = df_21[df_21["nationality_name"].isin(nationality_filter)]
        df_22 = df_22[df_22["nationality_name"].isin(nationality_filter)]

    if position_filter:
        df_21 = df_21[df_21["player_positions"].isin(position_filter)]
        df_22 = df_22[df_22["player_positions"].isin(position_filter)]

    col1, col2 = st.columns(2)

    with col1:
        try:
            st.markdown("### 2021 Player Stats")
            st.markdown("**Top 10 Players by Overall Rating (2021):**")
            st.table(df_21.sort_values("overall", ascending=False).head(10)[["short_name", "club_name", "overall", "value_eur"]])

            rating_bins_21 = pd.cut(df_21["overall"], bins=10)
            rating_dist_21 = rating_bins_21.value_counts().sort_index()
            rating_dist_21.index = rating_dist_21.index.astype(str)
            st.bar_chart(rating_dist_21)

            value_bins_21 = pd.cut(df_21["value_eur"], bins=10)
            value_dist_21 = value_bins_21.value_counts().sort_index()
            value_dist_21.index = value_dist_21.index.astype(str)
            st.bar_chart(value_dist_21)
        except ValueError:
            st.error("The data does not exist")
            st.stop()
            

    with col2:
        try:
            st.markdown("### 2022 Player Stats")
            st.markdown("**Top 10 Players by Overall Rating (2022):**")
            st.table(df_22.sort_values("overall", ascending=False).head(10)[["short_name", "club_name", "overall", "value_eur"]])
    
            rating_bins_22 = pd.cut(df_22["overall"], bins=10)
            rating_dist_22 = rating_bins_22.value_counts().sort_index()
            rating_dist_22.index = rating_dist_22.index.astype(str)
            st.bar_chart(rating_dist_22)
    
            value_bins_22 = pd.cut(df_22["value_eur"], bins=10)
            value_dist_22 = value_bins_22.value_counts().sort_index()
            value_dist_22.index = value_dist_22.index.astype(str)
            st.bar_chart(value_dist_22)
    
        st.subheader("Average Comparison Between Years")
        avg_rating_21 = df_21["overall"].mean()
        avg_rating_22 = df_22["overall"].mean()
        avg_value_21 = df_21["value_eur"].mean()
        avg_value_22 = df_22["value_eur"].mean()
    
        avg_df = pd.DataFrame({
            "Year": ["2021", "2022"],
            "Avg Rating": [avg_rating_21, avg_rating_22],
            "Avg Market Value (€)": [avg_value_21, avg_value_22]
        })
    
        st.dataframe(avg_df.style.format({"Avg Market Value (€)": "€{:.0f}", "Avg Rating": "{:.2f}"}))
    
        # Bar charts for average comparisons
        st.markdown("### Average Stats Comparison")
        st.markdown("**Average Rating:**")
        st.bar_chart(pd.DataFrame({
            "Year": ["2021", "2022"],
            "Avg Rating": [avg_rating_21, avg_rating_22]
        }).set_index("Year"))
    
        st.markdown("**Average Market Value (€):**")
        st.bar_chart(pd.DataFrame({
            "Year": ["2021", "2022"],
            "Avg Market Value (€)": [avg_value_21, avg_value_22]
        }).set_index("Year"))
    except ValueError:
        st.error("The data does not exist")
        st.stop()



