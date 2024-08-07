import pandas as pd
import uvicorn
from fastapi import FastAPI
import joblib

app = FastAPI()
model  = joblib.load('multioutput_classifier.pkl')

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
        "substance_abuse":substance_abuse

    }
    























if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, reload=True)