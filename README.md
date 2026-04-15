# athlete-injury-prediction

A 4-week mini project from the Biotech Club at UT Dallas that predicts the injury risk of collegiate athletes based on factors like age, ACL risk score, training intensity, games per week, and more.

## What I Learned
Through this project, I learned more about machine learning and data science; throughout the project, learned:
* How to use prediction models like Logistic Regression and Decision Trees, and when to apply each
* How to work with DataFrames using Pandas
* How to visualize data using matplotlib and seaborn
* How to build, train, and test a machine learning model

## Running the Model
Prerequisites:
*  install Python and the following libraries (using pip install):
    - pip install pandas numpy matplotlib seaborn scikit-learn joblib

Setup:
1. Clone or download this repository
2. Place your dataset file (collegiate_athlete_injury_dataset.csv) in the same folder as the notebook
3. Open Athlete_Injuries_ML.ipynb in Google Colab or Jupyter Notebook

Running the Model:
1. Run all cells from top to bottom (click "Run All" in Google Colab)
2. The notebook will automatically preprocess the data, train both models (Logistic Regression and Decision Tree), and select the best one

Making a Prediction:
* To predict injury risk for a new athlete, fill in their stats as a dictionary and call predict_injury_risk()
* Example:
athlete = {

  "Age": 20,
  
    "Height_cm": 180,
  
    "Weight_kg": 75,
  
    "Training_Intensity": 5,
  
    "Training_Hours_Per_Week": 8,
  
    "Recovery_Days_Per_Week": 3,

    "Match_Count_Per_Week": 1,
  
    "Rest_Between_Events_Days": 2,
  
    "Fatigue_Score": 3,
  
    "Performance_Score": 85,
  
    "Team_Contribution_Score": 80,
  
    "Load_Balance_Score": 80,

    "ACL_Risk_Score": 50
  
}

prob, risk = predict_injury_risk(best_model, athlete, X_train.columns, scaler, numerical_cols)


print("Probability:", prob)


print("Risk Level:", risk)  # Low Risk / Medium Risk / High Risk

### Note:
Risk is classified as:

* Low Risk — probability below 33%
* Medium Risk — probability between 33–66%
* High Risk — probability above 66%
