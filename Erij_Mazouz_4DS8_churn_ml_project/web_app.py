from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Path to the saved model
MODEL_PATH = "GBM_model.joblib"

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {str(e)}")


# Home route to display the form
@app.route("/")
def home():
    return render_template("index.html")


# Route to make predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from the form data
        features = [float(x) for x in request.form.values()]
        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)

        # Display the prediction result
        return render_template("result.html", prediction=prediction[0])

    except Exception as e:
        return jsonify({"error": f"Error making prediction: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True)
