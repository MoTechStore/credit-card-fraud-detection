from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
with open('fraud_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    try:
        input_data = [float(request.form[f'feature{i}']) for i in range(30)]  # Adjust the range for your features
    except ValueError:
        return "Invalid input. Please provide numerical values."

    # Reshape input for prediction (1 sample with 30 features)
    input_array = np.array(input_data).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)[0][1]  # Probability of being fraud

    # Return the result
    if prediction[0] == 1:
        return f"Transaction is Genuine with a probability of {probability:.2f}"
    else:
        return f"Transaction is Fraudulent with a probability of {1 - probability:.2f}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
