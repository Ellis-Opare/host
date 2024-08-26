'''
import pandas as pd
import uvicorn
from fastapi import FastAPI
import joblib

app = FastAPI()
model  = joblib.load('gbm3_model.pkl')

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

'''
import pandas as pd
import uvicorn
from fastapi import FastAPI
import joblib
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = FastAPI()
model = joblib.load('gbm3_model.pkl')

def calculate_severity(score):
    if score <= 1/3:
        return 'Mild'
    elif score <= 2/3:
        return 'Moderate'
    else:
        return 'Extreme'

@app.get('/')
async def hello():
    return "HELLO WORLD"

@app.post("/classify")
async def read_root(input: dict):
    df = pd.DataFrame(input, index=range(1))

    # Ensure the input features match the model's expected features
    predicted = model.predict(df.values)
    predicted = predicted.astype(float)  # Convert to standard Python float

    disorders = {
        "Depression": float(predicted[0][0]),
        "Schizophrenia": float(predicted[0][1]),
        "Acute_and_transient_psychotic_disorder": float(predicted[0][2]),
        "Delusional_Disorder": float(predicted[0][3]),
        "BiPolar1": float(predicted[0][4]),  # Exclude from severity
        "BiPolar2": float(predicted[0][5]),  # Exclude from severity
        "Generalized_Anxiety": float(predicted[0][6]),
        "Panic_Disorder": float(predicted[0][7]),
        "Specific_Phobia": float(predicted[0][8]),
        "Social_Anxiety": float(predicted[0][9]),
        "OCD": float(predicted[0][10]),
        "PTSD": float(predicted[0][11]),
        "Gambling": float(predicted[0][12]),
        "Substance_Abuse": float(predicted[0][13])
    }

    severity = {}
    for disorder, score in disorders.items():
        if disorder not in ["BiPolar1", "BiPolar2"]:
            severity[disorder] = calculate_severity(score)

    # Pie chart data
    pie_data = {k: v for k, v in disorders.items() if k not in ["BiPolar1", "BiPolar2"]}

    fig, ax = plt.subplots()
    ax.pie(pie_data.values(), labels=pie_data.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    # Save the pie chart to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return {
        "Disorders": disorders,
        "Severity": severity,
        "Pie_Chart": img_str
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, reload=True)

