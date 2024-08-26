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
from io import BytesIO
import base64

app = FastAPI()
model = joblib.load('gbm3_model.pkl')

@app.get('/')
async def hello():
    return "HELLO WORLD"

def calculate_severity(probability):
    if probability <= 1/3:
        return "Mild"
    elif 1/3 < probability <= 2/3:
        return "Moderate"
    else:
        return "Severe"

def generate_pie_chart(data):
    labels = [key for key, value in data.items() if value['probability'] > 0]
    sizes = [value['probability'] for key, value in data.items() if value['probability'] > 0]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return image_base64

@app.post("/classify")
async def classify(input: dict):
    df = pd.DataFrame([input])  # Ensure the input is in DataFrame format with feature names

    predicted = model.predict_proba(df)  # Get the predicted probabilities

    disorders = ["Depression", "Schizophrenia", "Acute_and_transient_psychotic_disorder", 
                 "Delusional_Disorder", "BiPolar1", "BiPolar2", "Generalized_Anxiety", 
                 "Panic_Disorder", "Specific_Phobia", "Social_Anxiety", "OCD", 
                 "PTSD", "Gambling", "substance_abuse"]

    response = {}
    
    for i, disorder in enumerate(disorders):
        probability = float(predicted[0][1])  # Use the probability of the positive class
        severity = calculate_severity(probability)
        response[disorder] = {"probability": probability, "severity": severity}

    pie_chart = generate_pie_chart(response)

    return {
        "classification": response,
        "pie_chart": pie_chart
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, reload=True)
