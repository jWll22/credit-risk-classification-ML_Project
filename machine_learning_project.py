import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report
                             )
import os
'''
import os is used to create a directory,
which where the graphs are stored
'''

plot_fig_savConfig = "output_plots"
os.makedirs(plot_fig_savConfig, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

''' Part A: Data Acquisition and Understanding '''
# Loading of dataset using pandas
df = pd.read_csv("credit_risk_dataset.csv")
print(df)

# Displaying the shape of the dataset:
print("\nDisplaying the shape of the dataset (rows, columns)\n", df.shape)

# Displaying the columns of the dataset:
print("\nColumn Names:\n", df.columns.tolist())

# Displaying the info of the dataset:
print("\nDataset information:\n", df.info())

# Using .describe in the dataset:
print("\nStatistical overview of the dataset:\n", df.describe())

# Identifying missing values:
print("\nMissing values per column:\n", df.isnull().sum())
print("\nMissing values as percentages:\n", round(df.isnull().sum() / len(df) * 100))

''' Part B: Exploratory Data Analysis '''
# Target Variable Distribution Graph:
plt.figure(figsize=(6, 5))
sns.countplot(x="loan_status", data=df, hue="loan_status", palette="Set2", legend=False)
plt.title("Distribution of Loan Status (0 = Low Risk, 1 = High Risk)")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig(f"{plot_fig_savConfig}/1_target_distribution.png")
plt.show()
'''
Interpretation: The classes are imbalanced - the majority of applicants (~78%)
repaid their loan (0), while only ~22% defaulted (1). This imbalance means
accuracy alone can be misleading, so precision/recall/F1 matter more later.
'''

# Correlation heatmap:
plt.figure(figsize=(9, 7))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig(f"{plot_fig_savConfig}/2_correlation_heatmap.png")
plt.show()
'''
Interpretation: loan_percent_income and loan_amnt show a positive relationship
with loan_status, meaning larger loans relative to income are linked to higher
default risk. person_income is negatively correlated with default, i.e. higher
earners default less often. No two features are so strongly correlated that
multicollinearity is an immediate concern, but PCA in Part D will confirm this.
'''

# Boxplots for outliers:
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.boxplot(y=df["person_age"], ax=axes[0, 0], color="skyblue")
axes[0, 0].set_title("Boxplot - Age")
sns.boxplot(y=df["person_income"], ax=axes[0, 1], color="lightgreen")
axes[0, 1].set_title("Boxplot - Income")
sns.boxplot(y=df["person_emp_length"], ax=axes[1, 0], color="pink")
axes[1, 0].set_title("Boxplot - Employment Length")
sns.boxplot(y=df["loan_amnt"], ax=axes[1, 1], color="khaki")
axes[1, 1].set_title("Boxplot - Loan Amount")
plt.tight_layout()
plt.savefig(f"{plot_fig_savConfig}/3_boxplots_outliers.png")
plt.show()

'''
Interpretation: Age has extreme outliers (a maximum of 144 years old, which is
impossible). Employment length also has an impossible outlier (123 years).
Income and loan amount both show a long right tail (a small number of very high
values) which is realistic for financial data but still needs to be controlled
so it doesn't distort scaling and model training later.
'''

# Histograms of selected numerical features
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.histplot(df["person_age"], bins=30, kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Age Distribution")
sns.histplot(df["person_income"], bins=30, kde=True, ax=axes[0, 1], color="seagreen")
axes[0, 1].set_title("Income Distribution")
sns.histplot(df["loan_amnt"], bins=30, kde=True, ax=axes[1, 0], color="indianred")
axes[1, 0].set_title("Loan Amount Distribution")
sns.histplot(df["loan_int_rate"].dropna(), bins=30, kde=True, ax=axes[1, 1], color="orange")
axes[1, 1].set_title("Interest Rate Distribution")
plt.tight_layout()
plt.savefig(f"{plot_fig_savConfig}/4_histograms.png")
plt.show()

'''
Interpretation: Age and income are both right-skewed, most applicants are
young adults with moderate income, with a small number of older/high-earning
outliers. Loan amount is also right-skewed. Interest rate is closer to a multi-model
distribution, which makes sense because interest rate
is largely determined by discrete loan_grade bands.
'''


'''Part C: Data Cleaning and Preprocessing'''

df_clean = df.copy() # df.copy creates a duplicate dataframe to ensure that it does not affect the original.

# Remove duplicates:
duplicate_count = df_clean.duplicated().sum()
print(f"Number of duplicate rows found: {duplicate_count}")
df_clean = df_clean.drop_duplicates()
print(f"Shape after removing duplicates: {df_clean.shape}")

# Treating obvious data-entry error outliers:
before_treatment = df_clean.shape[0]
df_clean = df_clean[(df_clean["person_age"] <= 100) & (df_clean["person_emp_length"] <= 60)
                     | (df_clean["person_emp_length"].isnull())]
print(f"Rows removed for impossible age/employment values: {before_treatment - df_clean.shape[0]}")

# Handling missing features (with justification):
df_clean["person_emp_length"] = df_clean["person_emp_length"].fillna(df_clean["person_emp_length"].median())
df_clean["loan_int_rate"] = df_clean.groupby("loan_grade")["loan_int_rate"] \
    .transform(lambda x: x.fillna(x.median()))
print("\nMissing Values after imputation:\n", df_clean.isnull().sum())

# Capping amounts for income, loan amount, and loan-percent:
'''This is done to treat outliers whilst ensuring no values get deleted; it also helps with two models such as KNN & PCA'''
def cap_outliers_iqr(frame, column):
    q1, q3 = frame[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    frame[column] = frame[column].clip(lower=lower, upper=upper)
    return frame

for col in ["person_income", "loan_amnt", "loan_percent_income"]:
    df_clean = cap_outliers_iqr(df_clean, col)

# Encode categorical variables:
# ordinal coding for loan_grade
grade_order = {g: i for i, g in enumerate(sorted(df_clean["loan_grade"].unique()))}
df_clean["loan_grade"] = df_clean["loan_grade"].map(grade_order)

# binary coding for cd_person_default_on_file:
df_clean["cb_person_default_on_file"] = df_clean["cb_person_default_on_file"].map({"Y": 1, "N": 0})

# Turns person_home_ownershio & loan_intent which contain categorical data; this is known as one-hot
df_clean = pd.get_dummies(df_clean, columns=["person_home_ownership", "loan_intent"],
                           drop_first=True)
print("\nColumns after encoding:\n", df_clean.columns.to_list())

# Separation of features, then split before scaling:
X = df_clean.drop(columns=["loan_status"])
y = df_clean["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"\nTraining set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

# Feature Scaling:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''Part D: Principal Component Analysis (PCA)'''

# Apply PCA to scaled data:
pca_full = PCA()
pca_full.fit(X_train_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained variance ratio per component:")
print(np.round(explained_variance, 4))
print("\nCumulative explained variance:")
print(np.round(cumulative_variance, 4))

# Plot for cumulative variance:
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.axhline(y=0.90, color="red", linestyle="--", label="90% variance")
plt.axhline(y=0.95, color="green", linestyle="--", label="95% variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance vs. Number of Components")
plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_fig_savConfig}/5_pca_cumulative_variance.png")
plt.show()

# Select several components capturing at least 90–95% variance:
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"\nNumber of components needed to reach >=90% variance: {n_components_90}")

# Create PCA-transformed dataset:
pca = PCA(n_components=n_components_90)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"\nShape of PCA-transformed training data: {X_train_pca.shape}")

'''Part E: Model Development (Before and After PCA)'''
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

results = []

def evaluate_model(model, X_tr, X_te, y_tr, y_te, model_name, feature_set_name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    predit = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred)

    print(f"\n--- {model_name} | {feature_set_name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {predit:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_te, y_pred))

    results.append({
        "Model": model_name, "Feature Set": feature_set_name,
        "Accuracy": acc, "Precision": predit, "Recall": rec, "F1-score": f1,
    })
    return model

# Train using original features:
for name, model in models.items():
    evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test,
                    name, "Original Features")
# Train using PCA-transformed features:
for name, model in models.items():
    fresh_model = type(model)(**model.get_params())
    evaluate_model(fresh_model, X_train_pca, X_test_pca, y_train, y_test,
                    name, "PCA Features")

# Comparison Table:
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

'''Part F: Model Optimization:'''
best_row = results_df.loc[results_df["F1-score"].idxmax()]
best_model_name = best_row["Model"]
best_feature_set = best_row["Feature Set"]
print(f"Best performing combination: {best_model_name} on {best_feature_set} "
      f"(F1-score = {best_row['F1-score']:.4f})")

if best_feature_set == "Original Features":
    X_tr_final, X_te_final = X_train_scaled, X_test_scaled
else:
    X_tr_final, X_te_final = X_train_pca, X_test_pca

# Hyperparameter grids for each possible best model
param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    },
    "K-Nearest Neighbours": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
}

base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbours": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

grid_search = GridSearchCV(
    estimator=base_models[best_model_name],
    param_grid=param_grids[best_model_name],
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
grid_search.fit(X_tr_final, y_train)

print(f"\nBest hyperparameters found: {grid_search.best_params_}")

tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(X_te_final)

tuned_acc = accuracy_score(y_test, y_pred_tuned)
tuned_prec = precision_score(y_test, y_pred_tuned)
tuned_rec = recall_score(y_test, y_pred_tuned)
tuned_f1 = f1_score(y_test, y_pred_tuned)

print("\nBefore tuning (baseline):\n")
print(best_row[["Accuracy", "Precision", "Recall", "F1-score"]])

print("\nAFTER tuning (GridSearchCV):\n")
print(f"Accuracy : {tuned_acc:.4f}")
print(f"Precision: {tuned_prec:.4f}")
print(f"Recall   : {tuned_rec:.4f}")
print(f"F1-score : {tuned_f1:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))
print("Classification Report:")
print(classification_report(y_test, y_pred_tuned))

"""
Impact on bias and variance:
GridSearchCV searches for the hyperparameter combination that generalises best 
on unseen folds (cross-validation), rather than the combination that simply
memorises the training data. If tuning increases test performance close to
training performance, that means variance (overfitting) has gone down. If a
very shallow/simple untuned model was underperforming on both train and test,
tuning that lets the model grow more complex (e.g. a deeper tree) reduces
bias (underfitting). In short: tuning aims to find the perfect balance on the
bias-variance trade-off rather than pushing purely for training accuracy.
"""
