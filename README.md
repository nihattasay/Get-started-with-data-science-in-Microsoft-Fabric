# Get-started-with-data-science-in-Microsoft-Fabric
# 🧪 Data Science in Microsoft Fabric

This repository demonstrates a complete machine learning workflow using **Microsoft Fabric**. You will ingest and explore data, engineer features, train both regression and classification models, and track experiments using **MLflow** — all within the Fabric environment.

## 🚀 What You'll Learn

- Creating a workspace in Microsoft Fabric
- Working with notebooks and Spark dataframes
- Using Data Wrangler for feature engineering
- Building regression and classification models using `scikit-learn`
- Tracking model experiments with MLflow
- Saving and managing trained models

---

## 🗂️ Lab Outline

### 1. **Workspace Setup**
Create a new workspace in Microsoft Fabric (Trial or Premium license required).

### 2. **Notebook Creation**
Start a new notebook and set up a markdown introduction:

```markdown
# Data science in Microsoft Fabric
3. Data Ingestion

Load the Diabetes dataset from Azure Open Datasets:

blob_account_name = "azureopendatastorage"
blob_container_name = "mlsamples"
blob_relative_path = "diabetes"
blob_sas_token = r""

wasbs_path = f"wasbs://{blob_container_name}@{blob_account_name}.blob.core.windows.net/{blob_relative_path}"
spark.conf.set(f"fs.azure.sas.{blob_container_name}.{blob_account_name}.blob.core.windows.net", blob_sas_token)

df = spark.read.parquet(wasbs_path)
display(df)
4. Data Visualization

Visualize the label distribution using a box plot on the Y column.
5. Data Preparation

Convert Spark DataFrame to Pandas and create a binary classification label:
df = df.toPandas()
df["Risk"] = (df["Y"] > 211.5).astype(int)
6. Feature Engineering

Launch Data Wrangler to explore and preprocess the data.
🤖 Model Training
📈 Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mlflow

X, y = df_clean[['AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6']], df_clean['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

mlflow.set_experiment("diabetes-regression")

with mlflow.start_run():
    mlflow.autolog()
    model = LinearRegression()
    model.fit(X_train, y_train)
🧠 Classification Model
from sklearn.linear_model import LogisticRegression

X, y = df_clean[['AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6']], df_clean['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

mlflow.set_experiment("diabetes-classification")

with mlflow.start_run():
    mlflow.sklearn.autolog()
    model = LogisticRegression(C=1/0.1, solver="liblinear")
    model.fit(X_train, y_train)
📊 Experiment Tracking

View and compare your regression and classification runs directly in Microsoft Fabric's MLflow UI.
💾 Save the Best Model

After reviewing experiment results, save the best-performing model:

    Select Save as ML model in the experiment pane.

    Name it model-diabetes.

    Use this model in downstream applications or scoring pipelines.

🧹 Clean Up

Once you're done, delete the workspace to avoid unnecessary resource usage.
