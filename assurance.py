# Import required modules
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
df = pd.read_csv("car_insurance.csv")
df = df.drop(columns=["id","gender", "education", "income", "married", "children", "postal_code"])
df['driving_experience'] = df['driving_experience'].replace({
    '0-9y': 0,
    '10-19y': 1,
    '20-29y': 2,
    '30y+': 3
})

df['age'] = df['age'].replace({
    '16-25': 0,
    '26-39': 1,
    '40-64': 2,
    '65+': 3
})
df['vehicle_type'] = df['vehicle_type'].replace({
    'sedan': 0,
    'sports car': 1
})
df['vehicle_year'] = df['vehicle_year'].replace({
    'before 2015': 0,
    'after 2015': 1
})
col= ['age', 'driving_experience', 'credit_score', 'vehicle_ownership',
                'vehicle_year', 'annual_mileage', 'vehicle_type',
                'speeding_violations', 'duis', 'past_accidents']
bestf=None
besta=0
for i in col:
    X=df[[i]]
    y = df["outcome"]
    # Build pipeline
    steps = [("imputation", SimpleImputer()), ("logisticRegression", LogisticRegression())]
    pipeline = Pipeline(steps)
    # Cross-validation
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    cv_results = cross_val_score(pipeline, X, y, cv=kf)
    a=np.mean(cv_results)
    if  a>besta:
        besta=a
        bestf=i
best_feature_df = pd.DataFrame({
    'best_feature': [bestf],
    'best_accuracy': [besta] })

print(best_feature_df)
