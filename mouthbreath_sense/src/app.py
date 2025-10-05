from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Import predict function from inference
from mouthbreath_sense.src.inference import predict


app = FastAPI(title="MouthBreathSense API")

class Sample(BaseModel):
    input: list  # expecting a list of floats or nested lists

@app.get("/")
def root():
    return {"message": "MouthBreathSense API is running"}

@app.post("/predict")
def run_prediction(sample: Sample):
    arr = np.array(sample.input, dtype=np.float32)
    result = predict(arr)
    return {"prediction": int(result)}
