# athlete-injury-prediction

What this is - a 4 week project (mini project from the Biotech Club at UT Dallas) that predicts the injury risk of athletes based off many different factors (i.e., age, ACL risk score, games per week, etc.)

What I learned - I learned a lot more about prediction models. I learned a lot about how to use different prediction models, such as logistic regression and decision trees, and when to use them. I learned how to use dataframes using Pandas, how to plot using matplotlib.pyplot, and what how to create and teach/test a machine learning model.

Prerequisites:
*  install Python and the following libraries (using pip install):
    - pip install pandas numpy matplotlib seaborn scikit-learn joblib

Setup:
1. Clone or download this repository
2. Place your dataset file (collegiate_athlete_injury_dataset.csv) in the same folder as the notebook
3. Open Athlete_Injuries_ML.ipynb in Google Colab or Jupyter Notebook

Running the Model
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
