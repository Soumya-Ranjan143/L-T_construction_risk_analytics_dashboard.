
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="L&T Project Intelligencee",
    page_icon="🏗️",
    layout="wide"
)

# ==================================================
# 2. FILE PATHS
# app.py is inside 1_Steps
# ==================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "Outputs"

DATA_PATH = OUTPUT_DIR / "cleaned_data.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
RESULTS_PATH = OUTPUT_DIR / "model_results.csv"
MODELS_PATH = OUTPUT_DIR / "trained_models.joblib"
PREPARED_DATA_PATH = OUTPUT_DIR / "prepared_data.joblib"

# Fallback model stored in the main project folder
PKL_MODEL_PATH = PROJECT_DIR / "construction_risk_model_for_L&T.pkl"

# ==================================================
# 3. LOAD DATASET
# ==================================================
if not DATA_PATH.exists():
    st.error(f"Cleaned dataset not found: {DATA_PATH}")
    st.info("Check that Outputs/cleaned_data.csv exists.")
    st.stop()

df = pd.read_csv(DATA_PATH)

if "Delay_Risk" not in df.columns:
    st.error("The dataset is missing the Delay_Risk column.")
    st.stop()

# ==================================================
# 4. LOAD TRAINED MODELS
# ==================================================
saved_models = {}
model = None

if MODELS_PATH.exists():
    loaded_object = joblib.load(MODELS_PATH)

    if isinstance(loaded_object, dict):
        saved_models = {
            key: value
            for key, value in loaded_object.items()
            if hasattr(value, "predict")
        }

        # Prefer Random Forest
        for key in [
            "forest_model",
            "random_forest_model",
            "random_forest",
            "Random Forest"
        ]:
            if key in saved_models:
                model = saved_models[key]
                break

        # Otherwise find a model with feature importance
        if model is None:
            for candidate in saved_models.values():
                classifier = getattr(candidate, "named_steps", {}).get(
                    "classifier"
                )
                if hasattr(classifier, "feature_importances_"):
                    model = candidate
                    break

        # Otherwise use the first saved prediction model
        if model is None and saved_models:
            model = next(iter(saved_models.values()))

    elif hasattr(loaded_object, "predict"):
        model = loaded_object

# Fallback to the standalone .pkl model
if model is None and PKL_MODEL_PATH.exists():
    model = joblib.load(PKL_MODEL_PATH)

if model is None:
    st.error(
        "No usable trained model was found. Expected one of:\n\n"
        f"- {MODELS_PATH}\n\n"
        f"- {PKL_MODEL_PATH}"
    )
    st.stop()

# ==================================================
# 5. LOAD OR GENERATE FEATURE IMPORTANCE
# ==================================================
if IMPORTANCE_PATH.exists():
    importance = pd.read_csv(IMPORTANCE_PATH)
else:
    importance = pd.DataFrame()

if (
    importance.empty
    or not {"Feature", "Importance"}.issubset(importance.columns)
):
    importance = pd.DataFrame()

    classifier = getattr(model, "named_steps", {}).get("classifier")
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")

    if (
        classifier is not None
        and preprocessor is not None
        and hasattr(classifier, "feature_importances_")
    ):
        try:
            feature_names = preprocessor.get_feature_names_out()

            importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance": classifier.feature_importances_
            }).sort_values("Importance", ascending=False)

        except Exception:
            importance = pd.DataFrame()

# ==================================================
# 6. LOAD OR CALCULATE MODEL RESULTS
# ==================================================
if RESULTS_PATH.exists():
    results = pd.read_csv(RESULTS_PATH)
else:
    results = pd.DataFrame()

# If the evaluation CSV is absent, calculate metrics using
# the saved test set and saved models.
if results.empty and PREPARED_DATA_PATH.exists():
    try:
        prepared = joblib.load(PREPARED_DATA_PATH)

        X_test = prepared["X_test"]
        y_test = prepared["y_test"]

        models_to_evaluate = saved_models.copy()

        if not models_to_evaluate:
            models_to_evaluate = {"Loaded Model": model}

        metric_rows = []

        for name, candidate in models_to_evaluate.items():
            predictions = candidate.predict(X_test)

            metric_rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test,
                    predictions,
                    pos_label="Yes",
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_test,
                    predictions,
                    pos_label="Yes",
                    zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test,
                    predictions,
                    pos_label="Yes",
                    zero_division=0
                )
            })

        results = pd.DataFrame(metric_rows)

    except Exception as error:
        st.warning(
            "Could not calculate model metrics from the saved "
            f"test data: {error}"
        )

# ==================================================
# 7. DASHBOARD HEADER AND NAVIGATION
# ==================================================
st.title("L&T Project Intelligence")
st.subheader("Construction Delay Risk Analytics")

st.caption(
    "Machine-learning prototype using synthetic construction data. "
    "Not an official L&T system."
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Exploratory Data Analysis",
        "Model Performance",
        "Feature Importance",
        "Predict Project Risk"
    ]
)

# ==================================================
# 8. PROJECT OVERVIEW
# ==================================================
if page == "Project Overview":
    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Projects", f"{len(df):,}")
    col2.metric("Dataset Columns", len(df.columns))
    col3.metric("Missing Values", int(df.isna().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Delay Risk Distribution")
    st.bar_chart(df["Delay_Risk"].value_counts())

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Summary")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Statistical Summary")
    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

    st.download_button(
        "Download Cleaned Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

# ==================================================
# 9. EXPLORATORY DATA ANALYSIS
# ==================================================
elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")

    st.subheader("Delay Risk Distribution")
    st.bar_chart(df["Delay_Risk"].value_counts())

    if "Project_Type" in df.columns:
        st.subheader("Projects by Type")
        st.bar_chart(df["Project_Type"].value_counts())

    if "Region" in df.columns:
        st.subheader("Projects by Region")
        st.bar_chart(df["Region"].value_counts())

    if "Project_Progress_Percent" in df.columns:
        st.subheader("Project Progress Distribution")

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(
            data=df,
            x="Project_Progress_Percent",
            bins=20,
            kde=True,
            ax=ax
        )
        ax.set_title("Distribution of Project Progress")
        st.pyplot(fig)
        plt.close(fig)

    category_charts = [
        ("Material_Availability", "Risk by Material Availability"),
        ("Equipment_Availability", "Risk by Equipment Availability"),
        ("Weather_Risk", "Risk by Weather Risk"),
        ("Project_Complexity", "Risk by Project Complexity")
    ]

    for column, heading in category_charts:
        if column in df.columns:
            st.subheader(heading)

            risk_table = pd.crosstab(
                df[column],
                df["Delay_Risk"],
                normalize="index"
            )
            st.bar_chart(risk_table)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(11, 7))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ==================================================
# 10. MODEL PERFORMANCE
# ==================================================
elif page == "Model Performance":
    st.header("Model Performance Comparison")

    if results.empty:
        st.warning(
            "Evaluation results are unavailable. Run Notebook 5 "
            "and save its evaluation metrics, or check that "
            "Outputs/prepared_data.joblib exists."
        )
    else:
        st.dataframe(results.round(3), use_container_width=True)

        if "Model" in results.columns:
            metric_columns = [
                column for column in
                ["Accuracy", "Precision", "Recall", "F1 Score"]
                if column in results.columns
            ]

            if metric_columns:
                st.subheader("Metric Comparison")
                st.bar_chart(
                    results.set_index("Model")[metric_columns]
                )

    st.info(
        "Metrics use a held-out test set when available. "
        "Results on synthetic data do not guarantee performance "
        "on real construction projects."
    )

# ==================================================
# 11. FEATURE IMPORTANCE
# ==================================================
elif page == "Feature Importance":
    st.header("Feature Importance")

    if importance.empty:
        st.warning(
            "Feature importance is unavailable. Check that the "
            "saved model is a Random Forest pipeline or that "
            "Outputs/feature_importance.csv exists."
        )
    else:
        top_features = (
            importance.nlargest(15, "Importance")
            .sort_values("Importance")
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            top_features["Feature"].astype(str),
            top_features["Importance"]
        )
        ax.set_xlabel("Importance")
        ax.set_title("Top 15 Model Features")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Complete Feature Importance")
        st.dataframe(importance, use_container_width=True)

# ==================================================
# 12. PROJECT RISK PREDICTION
# ==================================================
elif page == "Predict Project Risk":
    st.header("Predict Construction Delay Risk")

    st.write(
        "Enter project details to estimate the delay-risk category."
    )

    categorical_columns = [
        "Project_Type",
        "Region",
        "Material_Availability",
        "Equipment_Availability",
        "Weather_Risk",
        "Supplier_Delay",
        "Project_Complexity"
    ]

    missing_columns = [
        column for column in categorical_columns
        if column not in df.columns
    ]

    if missing_columns:
        st.error(f"Dataset is missing columns: {missing_columns}")
        st.stop()

    def categories(column):
        values = df[column].dropna().astype(str).unique().tolist()
        return sorted(values)

    if any(not categories(column) for column in categorical_columns):
        st.error("At least one categorical field has no available values.")
        st.stop()

    with st.form("project_prediction_form"):
        project_type = st.selectbox(
            "Project Type", categories("Project_Type")
        )
        region = st.selectbox(
            "Region", categories("Region")
        )
        duration = st.number_input(
            "Planned Duration (Months)",
            min_value=1,
            value=12
        )
        progress = st.slider(
            "Project Progress (%)",
            0,
            100,
            50
        )
        budget = st.number_input(
            "Budget (Million DZD)",
            min_value=0.1,
            value=100.0
        )
        cost_used = st.slider(
            "Budget Used (%)",
            0.0,
            200.0,
            50.0
        )
        workers = st.number_input(
            "Number of Workers",
            min_value=1,
            value=100
        )
        material = st.selectbox(
            "Material Availability",
            categories("Material_Availability")
        )
        equipment = st.selectbox(
            "Equipment Availability",
            categories("Equipment_Availability")
        )
        weather = st.selectbox(
            "Weather Risk",
            categories("Weather_Risk")
        )
        supplier_delay = st.selectbox(
            "Supplier Delay",
            categories("Supplier_Delay")
        )
        complexity = st.selectbox(
            "Project Complexity",
            categories("Project_Complexity")
        )
        incidents = st.number_input(
            "Safety Incidents",
            min_value=0,
            value=0
        )

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        project = pd.DataFrame([{
            "Project_Type": project_type,
            "Region": region,
            "Planned_Duration_Months": duration,
            "Project_Progress_Percent": progress,
            "Budget_Million_DZD": budget,
            "Cost_Used_Percent": cost_used,
            "Number_of_Workers": workers,
            "Material_Availability": material,
            "Equipment_Availability": equipment,
            "Weather_Risk": weather,
            "Supplier_Delay": supplier_delay,
            "Project_Complexity": complexity,
            "Safety_Incidents": incidents
        }])

        try:
            prediction = model.predict(project)[0]

            st.subheader("Prediction Result")

            if prediction == "Yes":
                st.error("Delay Risk Predicted: YES")
            else:
                st.success("Delay Risk Predicted: NO")

            if hasattr(model, "predict_proba"):
                probabilities = dict(
                    zip(
                        model.classes_,
                        model.predict_proba(project)[0]
                    )
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Delay Risk Probability",
                    f"{probabilities.get('Yes', 0):.1%}"
                )
                col2.metric(
                    "No Delay Risk Probability",
                    f"{probabilities.get('No', 0):.1%}"
                )

                st.caption(
                    "Probabilities are model estimates, not guarantees "
                    "of actual project outcomes."
                )

            st.subheader("Input Project Details")
            st.dataframe(project, use_container_width=True)

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.write(
                "Check that the saved model was trained using the "
                "same input columns and compatible preprocessing."
            )

# ==================================================
# 13. FOOTER
# ==================================================
st.divider()
st.caption(
    "Academic prototype using synthetic construction data. "
    "Not an official L&T system or a validated real-world "
    "forecasting tool."
)
