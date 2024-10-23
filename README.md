# Diabetes_Prediction_System

Overview

The Diabetes Prediction System is a Python-based application that utilizes machine learning to predict whether a person is likely to have diabetes based on various health parameters. This project uses a Support Vector Machine (SVM) algorithm to build the predictive model, which is trained on the well-known Pima Indians Diabetes Database.
Features

    User-friendly command-line interface for inputting health data.
    Real-time predictions based on user input.
    Displays training and testing accuracy of the model.
    Robust input handling to ensure valid numerical values.

Requirements

To run the application, you'll need the following Python packages:

    numpy
    pandas
    scikit-learn

You can install the required packages using pip:

bash

pip install numpy pandas scikit-learn

Additionally, make sure you have the dataset file named diabetes.csv in the same directory as the script. You can download the dataset from the UCI Machine Learning Repository.
Usage

    Clone the repository to your local machine:

    bash

git clone https://github.com/yourusername/diabetes-prediction-system.git

Navigate to the project directory:

bash

cd diabetes-prediction-system

Run the application:

bash

    python diabetes_prediction.py

    Follow the prompts to enter health parameters for prediction. The application will provide feedback on whether the individual is likely to have diabetes based on the input data.

Parameters

When prompted, enter the following health parameters:

    Pregnancies: Number of times pregnant
    Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    Blood Pressure: Diastolic blood pressure (mm Hg)
    Skin Thickness: Triceps skin fold thickness (mm)
    Insulin: 2-Hour serum insulin (mu U/ml)
    BMI: Body mass index (weight in kg/(height in m)^2)
    Diabetes Pedigree Function: Diabetes pedigree function (a value that reflects family history of diabetes)
    Age: Age (years)

Example

plaintext

Welcome to the Diabetes Prediction System
Training Accuracy: 0.85
Testing Accuracy: 0.78

Please enter the following details:
Pregnancies: 3
Glucose: 117
Blood Pressure: 72
Skin Thickness: 23
Insulin: 30
BMI: 32.0
Diabetes Pedigree Function: 0.3725
Age: 29
Result: The person is likely to have diabetes.
Do you want to make another prediction? (yes/no): no

License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    The dataset is sourced from the UCI Machine Learning Repository.
    Special thanks to the community for their resources and guidance in machine learning and data science.

