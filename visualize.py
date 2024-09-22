import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data.csv')

# Pie Chart for Target Variable Distribution
plt.figure(figsize=(10, 7))
plt.title('Pie Chart: Distribution of Pneumonia Cases')
df['has_pneumonia'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90, wedgeprops=dict(width=0.3))
plt.ylabel('')  # Remove ylabel for better appearance
plt.show()

# Bar Graph for Average Feature Values
plt.figure(figsize=(10, 7))
plt.title('Bar Chart: Average Feature Values')
feature_means = df[['age', 'temperature', 'blood_oxygen_level', 'respiratory_rate']].mean()
feature_means.plot(kind='bar', color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
plt.xlabel('Features')
plt.ylabel('Average Value')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Histogram for Age Distribution
plt.figure(figsize=(8, 6))
plt.title('Histogram: Age Distribution')
plt.hist(df['age'], bins=20, color='#ff9999', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# Histogram for Temperature Distribution
plt.figure(figsize=(8, 6))
plt.title('Histogram: Temperature Distribution')
plt.hist(df['temperature'], bins=20, color='#66b3ff', edgecolor='black')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# Histogram for Blood Oxygen Level Distribution
plt.figure(figsize=(8, 6))
plt.title('Histogram: Blood Oxygen Level Distribution')
plt.hist(df['blood_oxygen_level'], bins=20, color='#99ff99', edgecolor='black')
plt.xlabel('Blood Oxygen Level')
plt.ylabel('Frequency')
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# Histogram for Respiratory Rate Distribution
plt.figure(figsize=(8, 6))
plt.title('Histogram: Respiratory Rate Distribution')
plt.hist(df['respiratory_rate'], bins=20, color='#ffcc99', edgecolor='black')
plt.xlabel('Respiratory Rate')
plt.ylabel('Frequency')
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()
