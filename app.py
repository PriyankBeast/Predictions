from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize the app
app = Flask(__name__)

# Load the machine learning model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    month = int(data['month'])
    date = int(data['date'])
    hour = int(data['hour'])
    
    # Year is fixed to 2024
    year = 2024

    # Prepare the input for the model (change this based on how your model expects input)
    input_data = np.array([[year, month, date, hour]])

    # Make a prediction using the model
    prediction = model.predict(input_data)[0]

    # Return the prediction
    return jsonify({"prediction": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)
