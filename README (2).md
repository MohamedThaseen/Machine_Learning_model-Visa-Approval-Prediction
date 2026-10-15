# EasyVisa – US Work Visa Approval Prediction

A machine learning project that helps the Office of Foreign Labor Certification (OFLC) streamline and speed up the review of work visa applications by predicting whether a case is likely to be **certified** or **denied**.

## Context

Businesses in the United States depend on a steady pipeline of skilled talent, and the Immigration and Nationality Act allows foreign workers to be employed when there aren't enough qualified US workers available. The OFLC reviews every employer application to confirm this before granting certification.

In FY 2016 alone, OFLC processed **775,979 employer applications** covering **1,699,957 positions** — a 9% increase year-over-year. Manually reviewing each case is becoming increasingly unsustainable as volume grows.

## Objective

OFLC engaged **EasyVisa** to build a data-driven solution. As the data scientist on this project, the goals are to:

1. Build a classification model that facilitates and speeds up the visa approval process.
2. Identify the key drivers behind visa certification vs. denial, and recommend a suitable applicant profile for employers to target.

## Dataset

The dataset (`EasyVisa.csv`) contains **25,480 visa applications** across **12 columns**, combining attributes of the employee and the sponsoring employer.

| Column | Description |
|---|---|
| `case_id` | Unique ID of the visa application |
| `continent` | Employee's continent of origin |
| `education_of_employee` | Employee's education level |
| `has_job_experience` | Whether the employee has prior job experience (Y/N) |
| `requires_job_training` | Whether the employee requires job training (Y/N) |
| `no_of_employees` | Number of employees at the sponsoring company |
| `yr_of_estab` | Year the sponsoring company was established |
| `region_of_employment` | Intended US region of employment |
| `prevailing_wage` | Average wage for similar roles in that region, to prevent underpayment |
| `unit_of_wage` | Wage unit: Hourly, Weekly, Monthly, or Yearly |
| `full_time_position` | Whether the role is full-time (Y/N) |
| `case_status` | **Target** — whether the visa was Certified or Denied |

The raw data had no missing values and no duplicate records, but `no_of_employees` contained anomalous negative values that were treated during preprocessing.

## Approach

1. **Exploratory Data Analysis** – univariate and bivariate analysis of all features against `case_status` to surface patterns (e.g., education level, continent, wage, company age).
2. **Data Preprocessing** – anomaly handling for negative employee counts, feature engineering (e.g., bucketing `yr_of_estab` into quantile-based ranges), encoding of categorical variables, and a train / validation / test split.
3. **Model Building** – five classification algorithms (Decision Tree, Random Forest, Bagging Classifier, Gradient Boosting, AdaBoost) trained and compared on three versions of the data:
   - Original (imbalanced) data
   - SMOTE-oversampled data
   - Random-undersampled data
4. **Hyperparameter Tuning** – `RandomizedSearchCV` used to tune the top-performing models (AdaBoost and Gradient Boosting, on both oversampled and undersampled data), optimizing for recall.
5. **Model Selection** – performance compared across training, validation, and test sets to pick the most generalizable model.
6. **Feature Importance & Business Insights** – interpretation of the final model's drivers to produce actionable recommendations.

## Results

| Model | Validation Recall | Test Recall |
|---|---|---|
| Tuned AdaBoost (oversampled) | ~88% | — |
| **Tuned Gradient Boosting (oversampled)** | **~89.5%** | **~88.8%** |

The **Gradient Boosting Classifier trained on SMOTE-oversampled data** was selected as the final model — it delivered the highest and most consistent recall across training, validation, and test sets, indicating strong generalization with minimal overfitting.

### Key drivers of visa certification

- Candidates with a **Master's, Doctorate, or High School** education background are more likely to be certified.
- Employers established between **1997 and 2016** are associated with higher certification rates.
- Applicants from **Europe, Asia, and the Americas** show higher certification likelihood.
- Employment in the **Northeast, South, and West** regions is associated with higher certification rates.

## Recommendations

- Prioritize screening/processing for applications matching the higher-approval-likelihood profile identified above to speed up overall throughput.
- Use the model's recall-optimized predictions to flag high-confidence certifications for faster review, freeing up reviewer time for borderline cases.
- Periodically retrain the model as policy, wage, and labor-market conditions evolve.

## Tech Stack

- **Python**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Modeling**: scikit-learn (Decision Tree, Random Forest, Bagging, Gradient Boosting, AdaBoost), XGBoost
- **Imbalanced Data Handling**: imbalanced-learn (SMOTE, RandomUnderSampler)
- **Hyperparameter Tuning**: RandomizedSearchCV

## Repository Structure

```
.
├── Advanced_Machine_Learning_Project.ipynb   # Full analysis, modeling, and evaluation notebook
├── EasyVisa.csv                              # Dataset (not included — see Data Access below)
└── README.md
```

## Data Access

The dataset used in this project is not included in this repository. Place `EasyVisa.csv` in the project root (or update the file path in the notebook) before running.

## How to Run

1. Clone the repository and install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```
2. Place `EasyVisa.csv` in the project directory.
3. Open and run `Advanced_Machine_Learning_Project.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.

## Author

Mohamed Rifakath Ali A
