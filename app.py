import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class DiabetesPredictionApp:
    def __init__(self):
        # Load and prepare data
        self.load_data()
        self.prepare_model()

    def load_data(self):
        # Load the diabetes dataset
        self.diabetes_df = pd.read_csv('diabetes.csv')
        self.diabetes_mean_df = self.diabetes_df.groupby('Outcome').mean()

        # Split the data into input and target variables
        X = self.diabetes_df.drop('Outcome', axis=1)
        y = self.diabetes_df['Outcome']

        # Scale the input variables
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=1)

    def prepare_model(self):
        # Create and train the model
        self.model = svm.SVC(kernel='linear')
        self.model.fit(self.X_train, self.y_train)

        # Calculate accuracies
        train_y_pred = self.model.predict(self.X_train)
        test_y_pred = self.model.predict(self.X_test)
        self.train_acc = accuracy_score(train_y_pred, self.y_train)
        self.test_acc = accuracy_score(test_y_pred, self.y_test)

    def predict(self, input_data):
        # Scale the input data
        input_data_scaled = self.scaler.transform([input_data])

        # Make prediction
        prediction = self.model.predict(input_data_scaled)

        return prediction[0]

def main():
    app = DiabetesPredictionApp()

    print("Welcome to the Diabetes Prediction System")
    print(f"Training Accuracy: {app.train_acc:.2f}")
    print(f"Testing Accuracy: {app.test_acc:.2f}")

    while True:
        try:
            # Get user input
            print("\nPlease enter the following details:")
            pregnancies = float(input("Pregnancies: "))
            glucose = float(input("Glucose: "))
            blood_pressure = float(input("Blood Pressure: "))
            skin_thickness = float(input("Skin Thickness: "))
            insulin = float(input("Insulin: "))
            bmi = float(input("BMI: "))
            dpf = float(input("Diabetes Pedigree Function: "))
            age = float(input("Age: "))

            # Prepare input data
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, dpf, age]

            # Make prediction
            prediction = app.predict(input_data)

            # Display result
            if prediction == 1:
                print("Result: The person is likely to have diabetes.")
            else:
                print("Result: The person is unlikely to have diabetes.")

            # Check if the user wants to continue
            cont = input("Do you want to make another prediction? (yes/no): ")
            if cont.lower() != 'yes':
                break

        except ValueError:
            print("Please enter valid numerical values.")

if __name__ == "__main__":
    main()

