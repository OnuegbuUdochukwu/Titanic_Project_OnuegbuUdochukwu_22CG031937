import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# 1. Load Dataset
data = pd.read_csv('model/data.csv')

# 2. Feature Selection
# Selected features: Pclass, Sex, Age, Fare, Embarked
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'

X = data[features]
y = data[target]

# 3. Data Preprocessing Setup
numeric_features = ['Age', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Pclass', 'Sex', 'Embarked'] # Pclass is categorical/ordinal 
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Model Pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42))])

# 5. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate Model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Save Model
with open('model/titanic_survival_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved to model/titanic_survival_model.pkl")

# 8. Verification (Reload and Test)
with open('model/titanic_survival_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

sample = X_test.iloc[0:1]
prediction = loaded_model.predict(sample)
print(f"\nVerification Prediction for sample:\n{sample}")
print(f"Prediction: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
