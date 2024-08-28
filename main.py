'''
import pandas as pd
import uvicorn
from fastapi import FastAPI
import joblib

app = FastAPI()
model  = joblib.load('gbm4_model.pkl')

@app.get('/')
async def hello():
    return "HELLO WORLD"


@app.post("/classify")
async def read_root(input:dict):
    
    
    df = pd.DataFrame(input,index=range(1))

    predicted =  model.predict(df.values)
    Depression = str(predicted[0][0])
    Schizophrenia = str(predicted[0][1])
    Acute_and_transient_psychotic_disorder = str(predicted[0][2])
    Delusional_Disorder = str(predicted[0][3])
    BiPolar1 = str(predicted[0][4])
    BiPolar2 = str(predicted[0][5])
    Generalized_Anxiety = str(predicted[0][6])
    Panic_Disorder = str(predicted[0][7])
    Specific_Phobia = str(predicted[0][8])
    Social_Anxiety = str(predicted[0][9])
    OCD= str(predicted[0][10])
    PTSD= str(predicted[0][11])
    Gambling = str(predicted[0][12])
    substance_abuse = str(predicted[0][13])
    Others = str(predicted[0][14])
    


    return {
        "Depression":Depression,
        "Schizophrenia":Schizophrenia,
        "Acute_and_transient_psychotic_disorder":Acute_and_transient_psychotic_disorder,
        "Delusional_Disorder":Delusional_Disorder,
        "BiPolar1":BiPolar1,
        "BiPolar2":BiPolar2,
        "Generalized_Anxiety":Generalized_Anxiety,
        "Panic_Disorder":Panic_Disorder,
        "Specific_Phobia":Specific_Phobia,
        "Social_Anxiety":Social_Anxiety,
        "OCD":OCD,
        "PTSD": PTSD,
        "Gambling":Gambling,
        "substance_abuse":substance_abuse,
        "Others":Others

    }
    
@app.post("/predict_proba")
async def predict_proba(input: dict):
    df = pd.DataFrame(input, index=range(1))

    # Ensure the input features match the model's expected features
    proba = model.predict_proba(df.values)

    # Assuming it's a multi-output model, probabilities will be a list of arrays
    disorder_probabilities = {
        "Depression": float(proba[0][0][1]),
        "Schizophrenia": float(proba[1][0][1]),
        "Acute_and_transient_psychotic_disorder": float(proba[2][0][1]),
        "Delusional_Disorder": float(proba[3][0][1]),
        "BiPolar1": float(proba[4][0][1]),
        "BiPolar2": float(proba[5][0][1]),
        "Generalized_Anxiety": float(proba[6][0][1]),
        "Panic_Disorder": float(proba[7][0][1]),
        "Specific_Phobia": float(proba[8][0][1]),
        "Social_Anxiety": float(proba[9][0][1]),
        "OCD": float(proba[10][0][1]),
        "PTSD": float(proba[11][0][1]),
        "Gambling": float(proba[12][0][1]),
        "Substance_Abuse": float(proba[13][0][1]),
        "Others": float(proba[14][0][1])
    }

    return {
        "Probabilities": disorder_probabilities
    }



















if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, reload=True)

'''
import pandas as pd
import uvicorn
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('gbm4_model.pkl')

@app.get('/')
async def hello():
    return "HELLO WORLD"

@app.post("/classify")
async def classify(input: dict):
    df = pd.DataFrame(input, index=range(1))

    predicted = model.predict(df.values)
    response = {
        "Depression": predicted[0][0],
        "Schizophrenia": predicted[0][1],
        "Acute_and_transient_psychotic_disorder": predicted[0][2],
        "Delusional_Disorder": predicted[0][3],
        "BiPolar1": predicted[0][4],
        "BiPolar2": predicted[0][5],
        "Generalized_Anxiety": predicted[0][6],
        "Panic_Disorder": predicted[0][7],
        "Specific_Phobia": predicted[0][8],
        "Social_Anxiety": predicted[0][9],
        "OCD": predicted[0][10],
        "PTSD": predicted[0][11],
        "Gambling": predicted[0][12],
        "Substance_Abuse": predicted[0][13],
        "Others": predicted[0][14]
    }

    return response

@app.post("/severity")
async def severity(input: dict):
    df = pd.DataFrame(input, index=range(1))

    # Get model predictions
    predicted = model.predict(df.values)[0]

    # Define the number of features for each disorder
    total_features = {
        "Depression": 13,
        "Schizophrenia": 9,
        "Acute_and_transient_psychotic_disorder": 6,
        "Delusional_Disorder": 7,
        "BiPolar1": 8,
        "BiPolar2": 7,
        "Generalized_Anxiety": 7,
        "Panic_Disorder": 4,
        "Specific_Phobia": 7,
        "Social_Anxiety":12,
        "OCD": 4,
        "PTSD": 7,
        "Gambling": 5,
        "Substance_Abuse": 8,
        "Others": 10
    }

    severity_response = {}

    for disorder, present in zip(total_features.keys(), predicted):
        if present:
            features_present = sum(df.iloc[0].values)
            total = total_features[disorder]
            ratio = features_present / total

            if ratio <= 1/3:
                severity = 'Mild'
            elif 1/3 < ratio <= 2/3:
                severity = 'Moderate'
            else:
                severity = 'Extreme'

            severity_response[disorder] = severity

    return severity_response

@app.post("/predict_proba")
async def predict_proba(input: dict):
    df = pd.DataFrame(input, index=range(1))

    # Ensure the input features match the model's expected features
    proba = model.predict_proba(df.values)

    # Assuming it's a multi-output model, probabilities will be a list of arrays
    disorder_probabilities = {
        "Depression": float(proba[0][0][1]),
        "Schizophrenia": float(proba[1][0][1]),
        "Acute_and_transient_psychotic_disorder": float(proba[2][0][1]),
        "Delusional_Disorder": float(proba[3][0][1]),
        "BiPolar1": float(proba[4][0][1]),
        "BiPolar2": float(proba[5][0][1]),
        "Generalized_Anxiety": float(proba[6][0][1]),
        "Panic_Disorder": float(proba[7][0][1]),
        "Specific_Phobia": float(proba[8][0][1]),
        "Social_Anxiety": float(proba[9][0][1]),
        "OCD": float(proba[10][0][1]),
        "PTSD": float(proba[11][0][1]),
        "Gambling": float(proba[12][0][1]),
        "Substance_Abuse": float(proba[13][0][1]),
        "Others": float(proba[14][0][1])
    }

    return {
        "Probabilities": disorder_probabilities
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, reload=True)


