from sqlmodel import SQLModel, Field
from typing import Optional

from datetime import datetime

class CreateUpdateChurn(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    CreditScore: int
    #Geography: str
    #Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    prediction: str
    prediction_time: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    client_ip: str


class Churn(SQLModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

    class Config:
        schema_extra = {
            "example": {
                "CreditScore": 619,
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.00,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88,
            }
        }
