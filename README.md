# Decoding the Silent Resignation: Predicting Employee Job Change Intent

This project analyzes a dataset of employee information to build a predictive model that identifies which individuals are looking for a new job. In an era of "The Great Resignation" and "quiet quitting," understanding the drivers of employee turnover is crucial for companies aiming to improve retention and talent management.

This repository contains a complete data science workflow, from data cleaning and exploratory analysis to model building, evaluation, and feature importance analysis. The project culminates in identifying the key predictors that signal an employee's intent to seek new opportunities.

## Key Findings & Insights

*   **Top Predictor:** The **City Development Index** is the single most important feature in predicting whether an employee is looking for a job. This suggests that macro-economic factors and living conditions in an employee's city play a significant role in their career decisions.

*   **The Experience Factor:** Employees with very little experience (`<1` year) or a great deal of experience (`>20` years) are more likely to be looking for a new job compared to those in the middle of their careers.

*   **Class Imbalance is Key:** The dataset is highly imbalanced, with significantly more employees *not* looking for a job than those who are. Addressing this imbalance (e.g., using `scale_pos_weight` in XGBoost) was critical for building a useful model.

*   **Model Performance:** Both Random Forest and XGBoost performed well, but the tuned XGBoost model showed a slight edge, particularly in its ability to correctly identify employees who are looking for a job (higher recall for the positive class).

## Dataset

The project uses the `aug_train.csv` dataset, which contains various features about employees, including:

*   **Demographics:** `city`, `gender`, `education_level`, `major_discipline`
*   **Professional Experience:** `relevent_experience`, `experience`, `company_size`, `company_type`, `last_new_job`
*   **Metrics:** `city_development_index`, `training_hours`
*   **Target Variable:** `target` (0 = Not looking for a job change, 1 = Looking for a job change)

## Methodology

The project follows a structured data science pipeline:

### Data Cleaning & Preprocessing
*   Loaded the dataset and performed initial inspection to understand data types and identify missing values.
*   Cleaned categorical columns like `experience` and `last_new_job` by converting string values (e.g., `'>20'`, `'<1'`, `'never'`) into a numerical format for analysis and modeling.

### Exploratory Data Analysis (EDA)
*   Visualized the distribution of the target variable to confirm the class imbalance.
*   Created plots to analyze the relationship between key features (`training_hours`, `experience`, `last_new_job`) and an employee's intent to change jobs.

### Feature Engineering
*   Applied a mix of encoding strategies to handle categorical variables effectively:
    *   **Ordinal Encoding:** For features with an inherent order (e.g., `education_level`, `company_size`).
    *   **Label Encoding:** For binary features (e.g., `relevent_experience`).
    *   **One-Hot Encoding:** For nominal features with no inherent order (e.g., `city`, `gender`, `company_type`).

### Model Building & Evaluation
*   Split the data into training (80%) and testing (20%) sets.
*   Scaled numerical features using `StandardScaler` to ensure models were not biased by feature magnitudes.
*   Trained and compared two powerful ensemble models:
    *   **Random Forest Classifier:** With `class_weight='balanced'` to handle imbalance.
    *   **XGBoost Classifier:** With `scale_pos_weight` to give more importance to the minority class.
*   Evaluated models using Accuracy, ROC AUC Score, and a detailed Classification Report (Precision, Recall, F1-Score). Confusion matrices were also generated to visualize performance.

### Feature Importance Analysis
*   Used the best-performing model (XGBoost) to extract and visualize the most important features that contribute to the prediction.

## Results

Both models performed well, but XGBoost demonstrated superior performance in identifying the employees who were actually looking for a job change.

| Model                  | Accuracy | ROC AUC | Recall (Class 1) |
| :--------------------- | :------: | :-----: | :--------------: |
| **XGBoost Classifier** |  79.1%   |  81.8%  |     **77%**      |
| Random Forest          |  78.5%   |  81.3%  |       79%        |

The XGBoost model's higher recall for Class 1 (employees looking for a job) makes it more valuable in a real-world scenario, as the primary goal is to identify these individuals for retention efforts.

## How to Use This Repository

To run this project locally, follow these steps:

1.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```

2.  **Run the Jupyter Notebook:**
    *   Launch Jupyter Notebook or JupyterLab.
    *   Open `Decoding_the_Silent_Resignation_with_Outputs.ipynb` and run the cells.

## Future Work

*   **Hyperparameter Tuning:** Use techniques like GridSearchCV or Optuna to find the optimal hyperparameters for the models.
*   **Advanced Imputation:** Explore more sophisticated methods for handling missing values, such as KNNImputer or MICE.
*   **Alternative Models:** Experiment with other models like LightGBM, CatBoost, or a stacked ensemble to potentially improve performance further.
*   **Feature Creation:** Engineer new features from existing ones (e.g., ratios or interaction terms) to capture more complex relationships in the data.
