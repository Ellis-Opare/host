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
    


# Function to get user inputs
def get_user_inputs():
    """
    Prompts the user for inputs and returns them as a dictionary.
    """
    inputs = {}
    
    # Ask the user for each input
    inputs["Gender"] = input("Gender? (Male/Female/Other): ")
    inputs["Age"] = int(input("Age?: "))
    inputs["Depression or Hopelessness"] = int(input("Depression or Hopelessness? (0 for No, 1 for Yes): "))
    inputs["Previous Suicide Attempts"] = int(input("Previous Suicide Attempts? (0 for No, 1 for Yes): "))
    inputs["Alcohol or Drug Use"] = int(input("Alcohol or Drug Use? (0 for No, 1 for Yes): "))
    inputs["Rational Thinking Loss/Psychosis"] = int(input("Rational Thinking Loss/Psychosis? (0 for No, 1 for Yes): "))
    inputs["Separated, Divorced or Widowed"] = int(input("Separated, Divorced or Widowed? (0 for No, 1 for Yes): "))
    inputs["Organized Plan"] = int(input("Organized Plan? (0 for No, 1 for Yes): "))
    inputs["Social Supports"] = int(input("Social Supports? (0 for No, 1 for Yes): "))
    inputs["Stated Future Attempt to Harm Self"] = int(input("Stated Future Attempt to Harm Self? (0 for No, 1 for Yes): "))

    return inputs


# Mock function to simulate the detection of depression
def detect_depression(inputs):
    """
    Simulate a model detecting depression based on provided output.
    """
    # Assuming the detection is based on 'Depression or Hopelessness'
    return inputs.get("Depression or Hopelessness", 0) == 1


# Function to calculate probability
def calculate_probability(inputs):
    """
    Calculate the probability based on input criteria.
    The function assumes inputs are binary (0 for No, 1 for Yes).
    """
    # Map questions to their respective weights (can adjust as needed)
    weights = {
        "Depression or Hopelessness": 0.2,
        "Previous Suicide Attempts": 0.3,
        "Alcohol or Drug Use": 0.1,
        "Rational Thinking Loss/Psychosis": 0.15,
        "Separated, Divorced or Widowed": 0.1,
        "Organized Plan": 0.25,
        "Social Supports": -0.1,
        "Stated Future Attempt to Harm Self": 0.35
    }

    # Calculate weighted sum
    score = sum([inputs[key] * weights[key] for key in weights])

    # Determine probability range
    probability = min(max(score, 0), 1) * 100

    return probability


def main():
    # Get user inputs
    user_inputs = get_user_inputs()

    # Check if depression is detected by the model
    if detect_depression(user_inputs):
        # Call the function to calculate probability
        probability = calculate_probability(user_inputs)

        # Determine the probability category and print the result
        if probability >= 80:
            print("High probability of suicidal thoughts (≥80%)")
        elif 20 <= probability < 80:
            print("Intermediate probability of suicidal thoughts (20-79%)")
        else:
            print("Low probability of suicidal thoughts (≤19%)")
    else:
        print("No depression detected by the model.")





















if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, reload=True)
