from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_PATH = "GBM_model.pkl"

# Load the model at the start of the application
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    logger.error(f"Error loading model from {MODEL_PATH}: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


class PredictionInput(BaseModel):
    features: list


@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Prepare input data for prediction
        input_array = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(input_array)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


class RetrainInput(BaseModel):
    n_estimators: int = 100
    max_depth: int = None
    random_state: int = 42


def load_data():
    # Replace with your actual data loading process
    X_train = np.random.rand(100, 9)  # Example random data (replace with real dataset)
    y_train = np.random.randint(0, 2, 100)  # Random binary target variable
    return X_train, y_train


@app.post("/retrain")
def retrain(params: RetrainInput):
    """Endpoint to retrain the model with new hyperparameters."""
    try:
        global model
        X_train, y_train = load_data()

        # Create a new model with the provided parameters
        model = GradientBoostingClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            random_state=params.random_state,
        )

        # Train the model
        model.fit(X_train, y_train)

        # Save the retrained model
        joblib.dump(model, MODEL_PATH)

        return {"message": "Modèle réentraîné avec succès", "params": params.dict()}

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Erreur lors du réentraînement : {str(e)}"
        )
