# 📝 README: Crop Yield Prediction using Ensemble Models

This repository contains a Jupyter Notebook (`Crop_Yield_Prediction.ipynb`) demonstrating a comprehensive machine learning pipeline for predicting crop yield. The project covers data ingestion, preprocessing, feature engineering, model training with various ensemble methods, hyperparameter tuning, and performance evaluation, culminating in an interactive predictive interface.

## 📋 Table of Contents

1.  [**Project Overview**](#project-overview)
2.  [**Dataset Overview**](#dataset-overview)
3.  [**Methodology**](#methodology)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Feature Engineering](#feature-engineering)
    *   [Feature Selection](#feature-selection)
    *   [Model Training & Evaluation](#model-training--evaluation)
4.  [**Results Summary**](#results-summary)
5.  [**Best Performing Model**](#best-performing-model)
6.  [**Interactive Predictive Interface**](#interactive-predictive-interface)
7.  [**Dependencies**](#dependencies)
8.  [**Usage**](#usage)
9.  [**Next Steps**](#next-steps)

## 🌟 Project Overview

The goal of this project is to build a robust machine learning model to accurately predict crop yield based on various environmental and soil-related factors. By leveraging advanced ensemble techniques, we aim to provide insights into crop productivity and develop a tool that can assist in agricultural decision-making.

## 🌾 Dataset Overview

The project utilizes a synthetic dataset (`crop_yield_dataset.csv`) providing daily records of crop yield and related environmental factors over a 10-year period (2014–2023). Key features include:

| Feature           | Type        | Description                                                                                                                                                                                                                                                                                                                         |
| :---------------- | :---------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Date`            | DateTime    | Daily observations (dropped during preprocessing).                                                                                                                                                                                                                                                                                  |
| `Crop_Type`       | Categorical | Type of crop (e.g., Wheat, Corn, Rice).                                                                                                                                                                                                                                                                                             |
| `Soil_Type`       | Categorical | Soil classification (e.g., Sandy, Clay, Loamy).                                                                                                                                                                                                                                                                                     |
| `Soil_pH`         | Float       | Acidity/alkalinity of soil.                                                                                                                                                                                                                                                                                                         |
| `Temperature`     | Float (°C)  | Daily average temperature.                                                                                                                                                                                                                                                                                                          |
| `Humidity`        | Float (%)   | Average daily humidity.                                                                                                                                                                                                                                                                                                             |
| `Wind_Speed`      | Float (km/h)| Wind speed.                                                                                                                                                                                                                                                                                                                         |
| `N`, `P`, `K`     | Float (ppm) | Nitrogen, Phosphorus, Potassium levels.                                                                                                                                                                                                                                                                                             |
| `Crop_Yield`      | Float (t/ha)| **Target Variable:** Estimated daily yield per hectare.                                                                                                                                                                                                                                                                             |
| `Soil_Quality`    | Float (Index)| Index measuring soil health and fertility.

**Note:** The dataset is synthetic, and the yield model simplifies real-world complexities, such as biological stresses and regional variations.

## 🛠️ Methodology

### Data Preprocessing

1.  **Handling Missing Values:** Identified and removed rows where `Crop_Yield` was 0 (30.19% of the data), as these indicated a Missing At Random (MAR) pattern linked to extreme `Temperature` and `Humidity` values.
2.  **Categorical Encoding:** `Crop_Type` and `Soil_Type` features were encoded using `LabelEncoder` to convert them into numerical representations suitable for machine learning models.
3.  **Date Column Removal:** The `Date` column was dropped as its direct inclusion was not necessary for the chosen models.

### Feature Engineering

New features were created to capture more complex relationships and enhance predictive power:

*   **`NPK_Ratio`**: Ratio of Nitrogen to the sum of Phosphorus and Potassium ($$\frac{N}{P + K + \varepsilon}$$).
*   **`Soil_Nutrient_Score`**: Average of N, P, and K ($$\frac{N + P + K}{3}$$).
*   **`Temp_Humidity_Index`**: Interaction term between Temperature and Humidity ($$\text{Temperature} \times \text{Humidity}$$).

### Feature Selection

To identify the most influential variables and reduce dimensionality, a combination of techniques was applied:

*   **Correlation Analysis:** Both Pearson and Spearman correlations were used to understand linear and monotonic relationships between features and `Crop_Yield`.
*   **Random Forest Feature Importance:** A Random Forest Regressor was used to rank features by their importance in predicting crop yield.
*   **Recursive Feature Elimination (RFE):** RFE with a Random Forest estimator was applied to select the top 6 features: `Temperature`, `Soil_Quality`, `Temp_Humidity_Index`, `Crop_Type`, `Humidity`, and `NPK_Ratio`.

### Model Training & Evaluation

The data was split into an 80% training set and a 20% testing set. Features were scaled using `StandardScaler`. The following regression models were trained and evaluated:

*   Decision Tree Regressor
*   Random Forest Regressor (Baseline & Tuned)
*   XGBoost Regressor (Baseline & Tuned)
*   CatBoost Regressor (Baseline & Tuned)

Hyperparameter tuning was performed using `RandomizedSearchCV` for the ensemble models to optimize performance and prevent overfitting.

## 📈 Results Summary

Model performance was assessed using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² Score.

| **Model**                    |   **RMSE** |    **MAE** | **R² Score** |
| :--------------------------- | ---------: | ---------: | -----------: |
| Decision Tree                |     6.1941 |     4.3153 |       0.9241 |
| Random Forest (Baseline)     |     5.0723 |     3.5144 |       0.9491 |
| Random Forest (Tuned)        |     4.6541 |     3.2434 |       0.9571 |
| XGBoost (Baseline)           |     4.6527 |     3.2732 |       0.9572 |
| XGBoost (Tuned)              |     4.5899 |     3.2006 |       0.9583 |
| CatBoost (Baseline)          |     4.5496 |     3.1711 |       0.9590 |
| **CatBoost (Tuned)**         | **4.5180** | **3.1469** |   **0.9596** |

## ✅ Best Performing Model

The **Tuned CatBoost Regressor** emerged as the top-performing model, achieving the lowest RMSE and MAE, and the highest R² Score. It demonstrated robust predictive capabilities, explaining nearly 96% of the variance in crop yield.

## 🖥️ Interactive Predictive Interface

An interactive interface, built using `ipywidgets` within the notebook, allows users to input various parameters and receive real-time crop yield predictions from the tuned CatBoost model. 

**Inputs include:**
*   Temperature (°C)
*   Soil Quality (scaled 0–1)
*   Temperature-Humidity Index (%)
*   Crop Type (selected from a dropdown)
*   Humidity (%)
*   NPK Ratio

## 📦 Dependencies

To run this notebook, you will need the following Python libraries:

*   `pandas`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `xgboost`
*   `catboost`
*   `scipy`
*   `ipywidgets`
*   `joblib`

You can install them using pip:
```bash
pip install pandas scikit-learn matplotlib seaborn xgboost catboost scipy ipywidgets joblib
```

## 🚀 Usage

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/priyanshu-kr/CYP.git
    cd CYP
    ```
2.  **Download Dataset:** Ensure `crop_yield_dataset.csv` is in the same directory as the notebook. (The notebook includes `files.upload()` for Colab, but for local use, place the CSV manually).
3.  **Run the Notebook:** Open and run `Crop_Yield_Prediction.ipynb` in a Jupyter environment (e.g., Jupyter Lab, Google Colab).
4.  **Explore:** Follow the cells to see the data processing, model training, and interactive prediction interface.

## ⏭️ Next Steps

*   Deploy the tuned CatBoost model as a web service (e.g., using Flask or FastAPI).
*   Develop a user-friendly mobile or web application (e.g., with Streamlit, React) for broader access.
*   Integrate real-time data from satellite imagery and IoT sensors for dynamic predictions.
*   Further validate the model with diverse, real-world agricultural datasets.
