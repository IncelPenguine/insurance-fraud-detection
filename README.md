# Insurance Fraud Detection Model

This project establishes a robust machine learning pipeline designed to detect fraudulent insurance claims. It covers the end-to-end data science lifecycle: from in-depth data preprocessing and feature engineering to advanced modeling techniques, model evaluation with a focus on imbalanced data, and comprehensive explainability using SHAP. The work is consistently framed from a practical business and actuarial perspective, providing actionable insights for real-world insurance operations.

---

## Project Overview

Insurance fraud represents a significant financial burden for insurers, leading to increased premiums for policyholders and substantial losses for companies. This project directly addresses this critical challenge by developing a predictive model capable of classifying insurance claims as fraudulent or legitimate. Beyond mere classification, a core objective is to provide transparency and insight into the underlying factors that drive fraud-related decisions, empowering claims investigators and actuaries with data-driven intelligence.

---

## Objectives

This project aims to achieve the following key objectives:

* **Exploratory Data Analysis (EDA)**: Conduct a thorough EDA to understand data distributions, identify outliers, uncover trends, and reveal relationships within the insurance claims data.
* **Data Preprocessing & Cleaning**: Handle missing values, address inconsistencies, and prepare raw data for machine learning.
* **Feature Engineering**: Create domain-relevant, high-impact features such as `loss_ratio`, `claim_severity` per vehicle, `policy_age_years`, and `high_deductible` indicators, enhancing the model's predictive power.
* **Handle Class Imbalance**: Address the inherent imbalance in fraud datasets (minority class is fraud) using techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** and appropriate evaluation metrics.
* **Model Building & Evaluation**: Develop and evaluate multiple classification models, including **Logistic Regression**, **Random Forest Classifier**, and **XGBoost Classifier**, comparing their performance.
* **Hyperparameter Optimization**: Systematically tune the hyperparameters of the best-performing models (e.g., XGBoost) using techniques like `GridSearchCV` with **Stratified K-Fold Cross-Validation** to find optimal model configurations and ensure robust performance.
* **Model Interpretability (SHAP)**: Apply **SHAP (SHapley Additive exPlanations)** to interpret model predictions, identifying global feature importance and understanding individual prediction drivers.
* **Business & Actuarial Context**: Translate technical model outputs and insights into actionable business recommendations, aligning with actuarial principles for risk assessment and loss control.

---

## Dataset

The project utilizes a synthetic insurance claims dataset, simulating real-world complexities. Key characteristics include:

* **Size**: Approximately 1,000 records.
* **Features**: Around 40 features, encompassing:
    * **Customer Demographics**: Age, education level, occupation, relationship status, gender.
    * **Policy Information**: Policy bind date, state, annual premium, deductible, umbrella limit.
    * **Incident Details**: Incident type, severity, date, location, number of vehicles involved, property damage, police report availability, authorities contacted.
    * **Vehicle Details**: Auto make and model.
    * **Financial Details**: Capital gains, capital loss, total claim amount.
* **Target Variable**: A binary label `fraud_reported` ('Y' for fraud, 'N' for non-fraud), which exhibits significant class imbalance.
* **Data Quality**: The dataset contains initial missing values (represented by '?') and various categorical variables, necessitating comprehensive cleaning and encoding during the preprocessing pipeline.

---

## Workflow Summary

The project is structured into logical components for clarity and reproducibility:
```
insurance-fraud-detection/
├── data/
│   ├── raw/                  # Contains the original raw dataset (e.g., insurance_claims.csv)
│   └── processed/            # Stores cleaned, preprocessed, and feature-engineered datasets
│       ├── cleaned_insurance_data.csv    # Intermediate cleaned data
│       └── final_cleaned_insurance_data.csv # Final dataset ready for modeling
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis and Data Preprocessing
│   └── 02_Modeling.ipynb     # Model Training, Evaluation, and Interpretability
├── models/                  
├── reports/
│   └── Project_2.pdf         # Comprehensive project report (optional, can be a summary PDF of notebooks)
├── .gitignore                
├── README.md                 # Project description and guide (this file)
└── requirements.txt          # List of all Python dependencies
```

---

## Models and Results

Three distinct classification models were implemented and rigorously evaluated, with a strong emphasis on metrics critical for imbalanced fraud detection: **Recall** (to minimize missed fraud cases) and **F1-score** (to balance precision and recall).

| Model                   | Key Characteristics & Performance Insights                                                                                                                                                                                                            |
| :---------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | A linear baseline model. Showed lower **Recall** for the fraud class, indicating its limitations in capturing complex, non-linear patterns characteristic of fraud. Underperformed compared to tree-based models.                                   |
| **Random Forest** | An ensemble tree-based model. Demonstrated strong overall performance, particularly achieving competitive **Recall** scores, making it effective at identifying a higher proportion of actual fraudulent claims.                                      |
| **XGBoost Classifier** | A highly optimized gradient boosting framework. Exhibited the **best F1-score**, indicating the most balanced performance between precision and recall. This model was chosen for further optimization (hyperparameter tuning and SMOTE). |

Evaluation was consistently performed with the inherent class imbalance in mind, ensuring that metrics like Accuracy were de-emphasized in favor of Recall and F1-score, which are more relevant for fraud detection.

---

## Feature Importance with SHAP

**SHAP (SHapley Additive exPlanations)** was extensively used to interpret the black-box nature of our tree-based models (Random Forest and XGBoost), providing both global and local interpretability.

* **Global Feature Importance**: SHAP summary plots revealed the overall impact and direction of each feature on the model's output. Consistently, features such as `capital-gains`, `age`, and `policy_annual_premium` emerged as highly influential in fraud prediction.
* **Local Explanations (Beeswarm Plots)**: Beeswarm plots were utilized to visualize the individual impact of features on predictions across the dataset, showing how feature values drive predictions higher or lower.
* **Note on Other SHAP Plots**: Force plots and waterfall plots, while powerful, were intentionally excluded from direct rendering in the notebooks (or the final report) due to specific rendering and compatibility issues encountered with the `SHAP` library version used within the notebook environment, which sometimes leads to rendering failures in static outputs like PDFs. However, their underlying insights were still leveraged.

---

## Handling Class Imbalance

The raw dataset exhibited a significant class imbalance, with fraud claims constituting a minority (approximately 1:3 fraud-to-non-fraud ratio). To prevent model bias towards the majority class and improve fraud detection capabilities, the following strategies were employed:

* **SMOTE (Synthetic Minority Over-sampling Technique)**: Applied exclusively to the **training data** (`X_train`, `y_train`). SMOTE generates synthetic samples of the minority class (fraud) by interpolating between existing minority instances, thereby balancing the class distribution during model training. This helps the model learn more robust patterns for fraud.
* **Evaluation Metrics Focus**: The primary focus remained on metrics less sensitive to imbalance, such as **Recall** and **F1-score**, to accurately assess the model's performance in identifying fraud cases.
* **Stratified Cross-Validation**: `StratifiedKFold` was used during hyperparameter tuning and cross-validation to ensure that each fold maintains the original class distribution, providing more reliable performance estimates.

---

## Business and Actuarial Relevance

This project delivers a valuable machine learning solution with direct applicability and significant impact within the insurance sector:

* **Automated Claim Flagging**: The model can be integrated into the claims processing system to automatically flag incoming claims with a high probability of being fraudulent. This enables proactive intervention and reduces manual review for low-risk claims.
* **Investigator Prioritization**: By providing a fraud risk score and interpretable reasons (from SHAP), the model assists fraud investigation teams in prioritizing their efforts, focusing resources on the most suspicious claims.
* **Underwriting and Pricing Adjustments**: Insights from feature importance (e.g., specific policy details or incident characteristics driving fraud) can inform underwriting guidelines, potentially leading to more accurate risk assessment and premium adjustments for certain policy profiles.
* **Loss Ratio Improvement**: By effectively detecting and preventing fraudulent payouts, the model directly contributes to improving the insurer's overall **loss ratio**, enhancing profitability and financial stability.
* **Actuarial Insights**: From an actuarial perspective, the model supports advanced experience rating and credibility adjustments. The identification of fraud drivers can refine assumptions in claim cost models, leading to more accurate reserving and product pricing.

---

## Tools and Libraries

The project is developed using Python and relies on the following key libraries:

* **Python**: 3.10+
* **Data Manipulation**: `pandas`, `numpy`
* **Visualization**: `matplotlib`, `seaborn`
* **Machine Learning**: `scikit-learn` (for various models, preprocessing, model selection, and metrics), `xgboost`
* **Model Interpretability**: `shap`
* **Imbalance Handling**: `imbalanced-learn`

---

## Getting Started

To set up and run this project locally, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/insurance-fraud-detection.git](https://github.com/your-username/insurance-fraud-detection.git)
    cd insurance-fraud-detection
    ```
    *(Replace `your-username/insurance-fraud-detection.git` with your actual GitHub repository URL.)*

2.  **Install Dependencies**:
    It's highly recommended to use a virtual environment.
    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    # Activate the virtual environment
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate

    # Install required packages
    pip install -r requirements.txt
    ```

3.  **Download Raw Data**:
    * Place your `insurance_claims.csv` file into the `data/raw/` directory. (Ensure this matches the filename in your `01_EDA.ipynb` notebook).

4.  **Run the Notebooks**:
    Launch Jupyter Notebook or JupyterLab from the project's root directory:
    ```bash
    jupyter notebook
    ```
    Then, navigate to the `notebooks/` directory and run the notebooks in the following order:
    1.  `01_EDA.ipynb`: For data exploration, cleaning, and feature engineering. This will generate `cleaned_insurance_data.csv` and `final_cleaned_insurance_data.csv` in the `data/processed/` directory.
    
    2.  `02_Modeling.ipynb`: For model training, evaluation, hyperparameter tuning, imbalance handling, and model interpretability.

---

## Contact

For any questions or collaborations, feel free to reach out:

* **Gurpreet Singh**: incel.penguin@gmail.com