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

    proba = model.predict_proba(df.values)
    # Depression_proba = str(proba[0][0])
    # Schizophrenia_proba = str(proba[0][1])
    # Acute_and_transient_psychotic_disorder_proba = str(proba[0][2])
    # Delusional_Disorder_proba = str(proba[0][3])
    # BiPolar1_proba = str(proba[0][4])
    # BiPolar2_proba = str(proba[0][5])
    # Generalized_Anxiety_proba = str(proba[0][6])
    # Panic_Disorder_proba = str(proba[0][7])
    # Specific_Phobia_proba = str(proba[0][8])
    # Social_Anxiety_proba = str(proba[0][9])
    # OCD_proba = str(proba[0][10])
    # PTSD_proba = str(proba[0][11])
    # Gambling_proba = str(proba[0][12])
    # substance_abuse_proba = str(proba[0][13])
    # Others_proba = str(proba[0][14])
 
    


    return {
        "Disorder": {
            
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
            "Substance_Abuse":substance_abuse,
            "Others":Others,
        },

        "Probabilities":{

        "Depression": round(float(proba[0][0][1]),2),
        "Schizophrenia": round(float(proba[1][0][1]),2),
        "Acute_and_transient_psychotic_disorder": round(float(proba[2][0][1]),2),
        "Delusional_Disorder": round(float(proba[3][0][1]),2),
        "BiPolar1": round(float(proba[4][0][1]),2),
        "BiPolar2": round(float(proba[5][0][1]),2),
        "Generalized_Anxiety": round(float(proba[6][0][1]),2),
        "Panic_Disorder": round(float(proba[7][0][1]),2),
        "Specific_Phobia": round(float(proba[8][0][1]),2),
        "Social_Anxiety": round(float(proba[9][0][1]),2),
        "OCD": round(float(proba[10][0][1]),2),
        "PTSD": round(float(proba[11][0][1]),2),
        "Gambling": round(float(proba[12][0][1]),2),
        "Substance_Abuse": round(float(proba[13][0][1]),2),
        "Others": round(float(proba[14][0][1]),2),


        
        }

    }
