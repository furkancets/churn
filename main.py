from fastapi import FastAPI, Depends, Request
from models import Churn,CreateUpdateChurn
import os
from sqlalchemy.orm import Session
from mlflow.sklearn import load_model
from database import engine, get_db, create_db_and_tables

# Tell where is the tracking server and artifact server
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5001/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'

# Learn, decide and get model from mlflow model registry
model_name = "ChrunModeling"
model_version = 1
model = load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

app = FastAPI()

# Creates all the tables defined in models module
create_db_and_tables()


# Note that model will coming from mlflow
def makePrediction(model, request):
    # parse input from request
    CreditScore = request["CreditScore"]
    Age = request["Age"]
    Tenure = request["Tenure"]
    Balance = request["Balance"]
    NumOfProducts = request["NumOfProducts"]
    HasCrCard = request["HasCrCard"]
    IsActiveMember = request["IsActiveMember"]
    EstimatedSalary = request["EstimatedSalary"]
 

    # Make an input vector
    features = [[CreditScore,
                 Age,
                 Tenure,
                 Balance,
                 NumOfProducts,
                 HasCrCard,
                 IsActiveMember,
                 EstimatedSalary]]

    # Predict
    prediction = model.predict(features)

    return prediction[0]



# Insert Prediction information
def insertChurn(request, prediction, client_ip, db):
    newChurn = CreateUpdateChurn(
        #Geography = request["Geography"],
        #Gender = request["Gender"],
        Age = request["Age"],
        CreditScore = request["CreditScore"],
        Tenure = request["Tenure"],
        Balance = request["Balance"],
        NumOfProducts = request["NumOfProducts"],
        HasCrCard = request["HasCrCard"],
        IsActiveMember = request["IsActiveMember"],
        EstimatedSalary = request["EstimatedSalary"],
        prediction=prediction,
        client_ip=client_ip
    )

    with db as session:
        session.add(newChurn)
        session.commit()
        session.refresh(newChurn)

    return newChurn



# Electirical Price Prediction endpoint
@app.post("/churn/prediction")
async def predictPrice(request: Churn, fastapi_req: Request,  db: Session = Depends(get_db)):
    prediction = makePrediction(model, request.dict())
    db_insert_record = insertChurn(request=request.dict(), prediction=prediction,
                                          client_ip=fastapi_req.client.host,
                                          db=db)
    return {"prediction": prediction, "db_record": db_insert_record}
