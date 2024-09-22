import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv('data.csv')

# Check the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Data types of the columns
print(df.dtypes)

# Summary statistics of the numerical columns
print(df.describe())

# Check the distribution of the target variable
print(df['has_pneumonia'].value_counts())  # Assuming 100 has pneumonia and 100 people don't

# Add more noise to the features
df['age'] += np.random.normal(0, 20, size=len(df))  # Increase noise in age
df['temperature'] += np.random.normal(0, 2, size=len(df))  # Increase noise in temperature
df['blood_oxygen_level'] += np.random.normal(0, 5, size=len(df))  # Increase noise in blood oxygen level
df['respiratory_rate'] += np.random.normal(0, 3, size=len(df))  # Increase noise in respiratory rate

# Use multiple features (including temperature and respiratory_rate)
X = df[['age', 'temperature','blood_oxygen_level', 'respiratory_rate']]
y = df['has_pneumonia']

# Stratified train-test split to ensure balanced classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a wider parameter grid for the model
param_grid = {
    'model__C': [0.5, 1, 10],  # Higher regularization to force the model to underfit
    'model__solver': ['liblinear', 'lbfgs'],
}

# Create a pipeline with scaling, polynomial features, and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Add polynomial interactions
    ('model', LogisticRegression(random_state=42, max_iter=1000))  # Logistic Regression model
])

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=2)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("\nBest parameters found:")
print(grid_search.best_params_)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate the best model on the test data
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
# Get the predicted probabilities for the positive class
y_prob = best_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#### NEW DATA ####
# New test data with 5 entries
new_data = pd.DataFrame({
    'age': [55, 60, 50, 30, 45],  # Age of the new patients
    'temperature': [99.5, 100.0, 98.7, 102.3, 101.5],  # Temperature measurements
    'blood_oxygen_level': [74, 76, 72, 92, 88],  # Blood oxygen levels
    'respiratory_rate': [15, 16, 14, 18, 22]  # Respiratory rates
})

# Use the same pipeline (scaling, polynomial features, etc.) to predict on new data
new_predictions = best_model.predict(new_data)

# Display predictions
print("\nPredictions on new data:")
for i, pred in enumerate(new_predictions):
    print(f"New data point {i+1}: {'Has Pneumonia' if pred == 1 else 'No Pneumonia'}")




