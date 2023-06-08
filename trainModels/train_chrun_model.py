# import liblaries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

# Read dataset and split as train test
data = pd.read_csv("https://raw.githubusercontent.com/erkansirin78/datasets/master/Churn_Modelling.csv")
print(data.head())

data.drop(["RowNumber","CustomerId","Surname","Geography","Gender"],axis=1,inplace=True)
# Convert values to numeric

data["CreditScore"] = pd.to_numeric(data["CreditScore"], errors= 'coerce')
data["Age"] = pd.to_numeric(data["Age"], errors= 'coerce')
data["Tenure"] = pd.to_numeric(data["Tenure"], errors= 'coerce')
data["Balance"] = pd.to_numeric(data["Balance"], errors= 'coerce')
data["NumOfProducts"] = pd.to_numeric(data["NumOfProducts"], errors= 'coerce')
data["HasCrCard"] = pd.to_numeric(data["HasCrCard"], errors= 'coerce')
data["IsActiveMember"] = pd.to_numeric(data["IsActiveMember"], errors= 'coerce')
data["EstimatedSalary"] = pd.to_numeric(data["EstimatedSalary"], errors= 'coerce')
data["Exited"] = pd.to_numeric(data["Exited"], errors= 'coerce') 


# Delete all null values
data = data.dropna()


# Split the data as  X and y after train and test
X = data[["CreditScore", "Age", "Tenure", "Balance", 
          "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", 
           ]]
y = data["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.2, 
                                                random_state=42)
# # MLflow


# Determine the Urls
os.environ['MLFLOW_TRACKING_URI'] = 'http://host.docker.internal:5001/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://host.docker.internal:9000/'


# Eveluation metric method
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Set MLflow experiment
experiment_name = "Deploy Model new Churn Model with MinIO Mlflow"
mlflow.set_experiment(experiment_name)

registered_model_name="ChrunModeling"


# Determine number of model trees
number_of_trees=200

# Train model and register mlflow
with mlflow.start_run(run_name="with-reg-rf-sklearn") as run:
    estimator = RandomForestRegressor(n_estimators=number_of_trees)
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)

    (rmse, mae, r2) = eval_metrics(y_test, y_pred)

    print(f"Random Forest model number of trees: {number_of_trees}")
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("n_estimators", number_of_trees)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        mlflow.sklearn.log_model(estimator, "model", registered_model_name=registered_model_name)
    else:
        mlflow.sklearn.log_model(estimator, "model")

# # Optional Part


name = registered_model_name
client = MlflowClient()
# client.create_registered_model(name)


model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
print(model_uri)


mv = client.create_model_version(name, model_uri, run.info.run_id)
print("model version {} created".format(mv.version))
last_mv = mv.version
print(last_mv)

def print_models_info(models):
    for m in models:
        print("name: {}".format(m.name))
        print("latest version: {}".format(m.version))
        print("run_id: {}".format(m.run_id))
        print("current_stage: {}".format(m.current_stage))

def get_latest_model_version(models):
    for m in models:
        print("name: {}".format(m.name))
        print("latest version: {}".format(m.version))
        print("run_id: {}".format(m.run_id))
        print("current_stage: {}".format(m.current_stage))
    return m.version

models = client.get_latest_versions(name, stages=["None"])
print_models_info(models)

print(f"Latest version: { get_latest_model_version(models) }")
