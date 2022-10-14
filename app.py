from flask import Flask
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.sav', 'rb'))

listOfIds = [
    "gender",
    "senior",
    "partner",
    "dependents",
    "tenure",
    "service",
    "lines",
    "security",
    "backup",
    "protection",
    "support",
    "tv",
    "movies",
    "billing",
    "monthly",
    "total",
    "internet",
    "contract",
    "payment",
]

# Let's get the data from the form
def getChoices(listOfIds):
    choices = []
    ids = []
    for id in listOfIds:
        choice = float(request.form.get(id))
        choices.append(choice)
        ids.append(id)
    return choices, ids


# Here we will create one hot encoded features to add to our input
def getOneHotFeatures(choices):
    internetFeats = [0] * 2
    contractFeats = [0] * 2
    paymentFeats = [0] * 3

    internetChoice = int(choices[16])
    contractChoice = int(choices[17])
    paymentChoice = int(choices[18])

    if internetChoice != 0:
        internetFeats[internetChoice - 1] = 1

    if contractChoice != 0:
        contractFeats[contractChoice - 1] = 1

    if paymentChoice != 0:
        paymentFeats[paymentChoice - 1] = 1

    features = internetFeats + contractFeats + paymentFeats
    return features
    


# The making of the input vestor for the model
def constructInput():
    choices, _ = getChoices(listOfIds)
    oneHotFeatures = getOneHotFeatures(choices)
    
    input = choices[0:16] + oneHotFeatures

    return input

    


@app.route("/")
def home():
    return render_template("Form.html")

@app.route("/form", methods = ['GET'])
def form():
    return "Form"


@app.route("/result", methods = ['GET', 'POST'])
def result():
    if request.method == 'POST':
        features, ids = getChoices(listOfIds)
        input = constructInput()
        finalInput = [np.array(input)]
        prediction = model.predict(finalInput)

        output = round(prediction[0], 2)
        return render_template("Form.html", churn = output, features = input)
    return "No Data"


if __name__ == "__main__":
    app.run(debug=True)