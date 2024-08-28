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
        },

        # "Probabilities":{

        #     "Depression_proba":Depression_proba,
        #     "Schizophrenia_proba":Schizophrenia_proba,
        #     "Acute_and_transient_psychotic_disorder_proba":Acute_and_transient_psychotic_disorder_proba,
        #     "Delusional_Disorder_proba":Delusional_Disorder_proba,
        #     "BiPolar1_proba":BiPolar1_proba,  
        #     "BiPolar2_proba":BiPolar2_proba,
        #     "Generalized_Anxiety_proba":Generalized_Anxiety_proba,
        #     "Panic_Disorder_proba":Panic_Disorder_proba,
        #     "Specific_Phobia_proba":Specific_Phobia_proba, 
        #     "Social_Anxiety_proba":Social_Anxiety_proba,
        #     "OCD_proba":OCD_proba, 
        #     "PTSD_proba":PTSD_proba, 
        #     "Gambling_proba":Gambling_proba, 
        #     "substance_abuse_proba":substance_abuse_proba,
        #     "Others_proba":Others_proba,


        
        # }

    }
