from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
try:
    with open('model/titanic_survival_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file not found. Please run model/train_model.py first.")
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    
    if request.method == 'POST':
        try:
            # Get data from form
            pclass = int(request.form['pclass'])
            sex = request.form['sex']
            age = float(request.form['age'])
            fare = float(request.form['fare'])
            embarked = request.form['embarked'] # C, Q, S

            # Create DataFrame for prediction (must match training columns)
            input_data = pd.DataFrame([[pclass, sex, age, fare, embarked]], 
                                      columns=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked'])

            # Make prediction
            prediction = model.predict(input_data)[0]
            
            if prediction == 1:
                prediction_text = "Result: Survived 🟢"
            else:
                prediction_text = "Result: Did Not Survive 🔴"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
