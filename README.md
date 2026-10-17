# Construction Delay Risk Intelligence For L&T Group.

An AI-powered academic project that uses machine learning to analyze construction project data, identify potential delay risks, and present insights through an interactive dashboard.

## Project Overview

Construction projects can face delays due to weather conditions, material shortages, supplier delays, equipment availability, safety incidents, and project complexity. This project explores how machine learning can help identify potential delay risks based on these factors.

The project includes data analysis, preprocessing, model training, evaluation, feature importance analysis, and an interactive Streamlit dashboard.

**Note:** This is an independent academic prototype built using synthetic construction data. It is not an official Larsen & Toubro (L&T) product, and its predictions have not been validated for real-world project forecasting.

## Key Features

- Exploratory Data Analysis (EDA)
- Data cleaning and feature preparation
- Machine learning using Logistic Regression and Random Forest
- Model evaluation and performance comparison
- Feature importance analysis
- Construction delay risk prediction
- Interactive dashboard built with Streamlit
- Power BI dashboard for visual analytics

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Power BI
- **Dashboard:** Streamlit
- **Development Environment:** Jupyter Notebook

## Project Structure

```text
L&T Project Intelligence/
├── 1_Steps/
│   ├── 1_data understanding.ipynb
│   ├── 2_Exploratory Data Analysis (EDA).ipynb
│   ├── 3_Data cleaning and feature preparation.ipynb
│   ├── 4_Train the models.ipynb
│   ├── 5_Model Evaluation and Comparison.ipynb
│   ├── 6_Model Interpretation and Prediction.ipynb
│   ├── app.py
│   └── construction_risk_model_for_L&T.pkl
├── Outputs/
│   ├── cleaned_data.csv
│   ├── feature_importance.csv
│   ├── prepared_data.joblib
│   └── trained_models.joblib
├── India_Construction_Data_L&T.csv
├── README.md
└── LICENSE
```

*The listed artifacts may vary depending on which notebook steps have been executed.*

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Soumya-Ranjan143/L-T_construction_risk_analytics_dashboard.git
```

### 2. Open the project directory

```bash
cd L-T_construction_risk_analytics_dashboard
```

### 3. Install the required libraries

Make sure Python is installed, then run:

```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

## Run the Final Dashboard

The final dashboard is built using Streamlit. Run it from the directory containing `app.py`.

### Option A: If `app.py` is inside `1_Steps`

```bash
cd 1_Steps
python -m streamlit run app.py
```

### Option B: If `app.py` is in the project root

```bash
python -m streamlit run app.py
```

After starting the application, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

**Important:** The dashboard must load the model and data files from the correct locations. If it reports missing files, check that the required artifacts exist in `Outputs/` or update the paths in `app.py`. Run the required notebook steps first if those artifacts have not yet been generated.

## Notebook Execution Workflow

For a fresh setup, execute the notebooks in the following order:

1. **Data Understanding:** Inspect the dataset structure, columns, and data quality.
2. **Exploratory Data Analysis:** Explore distributions, relationships, and patterns.
3. **Data Cleaning and Feature Preparation:** Prepare features and split the data for training and testing.
4. **Train the Models:** Train the selected machine learning models.
5. **Model Evaluation and Comparison:** Compare model performance using evaluation metrics.
6. **Model Interpretation and Prediction:** Examine important features and test predictions.

Ensure that the notebooks save their required output files before launching the final dashboard.

## Target Variable

The target variable is `Delay_Risk`, with `Yes` and `No` labels. The models classify potential delay risk based on the available dataset; they do not directly predict the number of days a project will be delayed.

## Future Improvements

- Validate the approach using real, appropriately licensed construction data.
- Improve model generalization and probability calibration.
- Add project-level risk alerts and explainable predictions.
- Enhance dashboard usability and reporting.

## License

This project is distributed under the MIT License. Refer to the `LICENSE` file for the applicable terms.

## Author

**Soumya Ranjan Bhol**

Developed as an independent academic machine learning and data analytics project.
