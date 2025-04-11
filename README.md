🏦 Loan Approval Prediction
This project predicts whether a loan should be approved based on applicant data using machine learning algorithms. It employs Logistic Regression and Random Forest (with hyperparameter tuning) to analyze and make predictions on the dataset provided.

🌟 Features
Logistic Regression: A statistical model for binary classification problems.
Random Forest: A robust ensemble learning method that combines multiple decision trees for improved prediction accuracy.
Streamlit Application: An interactive web interface for testing and exploring loan approval predictions. https://ml-projects-ew5mkv5eutyhy4tpjmrgbn.streamlit.app/ 

💠 How to Use
Step 1: Prepare the Dataset
Name your dataset file as loan_data.csv.
Place it in the data/ folder of the repository.

Step 2: Train the Models
Run the following command in the terminal to train the machine learning models:
python train.py
This will process the dataset and save the trained models in the models/ folder.

Step 3: Launch the Streamlit Application
You can interact with the project using the Streamlit web interface. Use the following command to start the app:
streamlit run app.py

Alternatively, you can access the deployed Streamlit app directly through this link: Streamlit App Deployment. https://ml-projects-ew5mkv5eutyhy4tpjmrgbn.streamlit.app/


📂 Folder Structure
The repository is structured as follows:

data/: Contains the dataset (loan_data.csv). Any input data required for the project should be placed here.
models/: Stores the trained model files after the training process.
src/: Includes modular code for various tasks such as:
Preprocessing: Cleaning and preparing the data.
Training: Training the machine learning models.
Evaluation: Assessing the performance of the models.
Logging: Tracking and logging the training process.
app.py: The entry point for the Streamlit application. It defines the user interface and connects it with the trained models for interactive predictions.
train.py: The script responsible for training the machine learning models using the dataset in the data/ folder.


✅ Requirements
To set up the environment, follow these steps:

Install Python (if not already installed).

🔄 Summary of Workflow
Prepare: Place the dataset (loan_data.csv) in the data/ folder.
Train: Run train.py to train the models.
Launch: Use app.py to start the Streamlit application for interactive predictions.

