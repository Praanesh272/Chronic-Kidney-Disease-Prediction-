import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import tree

# Load and clean data
df_data = pd.read_csv('kidney_disease.csv')
df_data.drop('id', axis=1, inplace=True)

# Rename columns for consistency
df_data.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells',
                   'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea',
                   'serum_creatinine', 'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume',
                   'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
                   'coronary_artery_disease', 'appetite', 'peda_edema', 'anemia', 'class']

# Convert numeric columns with object types
text_columns = ['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']
for column in text_columns:
    df_data[column] = pd.to_numeric(df_data[column], errors='coerce')

# Impute missing numeric values with mean
num_cols = [col for col in df_data.columns if df_data[col].dtype != 'object']
for col in num_cols:
    df_data[col].fillna(df_data[col].mean(), inplace=True)

# Clean and impute categorical columns
df_data['diabetes_mellitus'] = df_data['diabetes_mellitus'].replace({' yes': 'yes', '\tyes': 'yes', '\tno': 'no'})
df_data['coronary_artery_disease'] = df_data['coronary_artery_disease'].replace({'\tno': 'no'})
df_data['class'] = df_data['class'].replace({'ckd\t': 'ckd'})

fill_values = {
    'red_blood_cells': 'normal',
    'pus_cell': 'normal',
    'pus_cell_clumps': 'notpresent',
    'bacteria': 'notpresent',
    'diabetes_mellitus': 'no',
    'coronary_artery_disease': 'no',
    'appetite': 'good',
    'peda_edema': 'no',
    'anemia': 'no'
}

for col, val in fill_values.items():
    df_data[col].fillna(val, inplace=True)

# Encode categorical columns
binary_mappings = {
    'red_blood_cells': {'normal': 1, 'abnormal': 0},
    'pus_cell': {'normal': 1, 'abnormal': 0},
    'pus_cell_clumps': {'present': 1, 'notpresent': 0},
    'bacteria': {'present': 1, 'notpresent': 0},
    'hypertension': {'yes': 1, 'no': 0},
    'diabetes_mellitus': {'yes': 1, 'no': 0},
    'coronary_artery_disease': {'yes': 1, 'no': 0},
    'appetite': {'good': 1, 'poor': 0},
    'peda_edema': {'yes': 1, 'no': 0},
    'anemia': {'yes': 1, 'no': 0},
    'class': {'ckd': 1, 'notckd': 0}
}

for col, mapping in binary_mappings.items():
    df_data[col] = df_data[col].map(mapping)

# Feature matrix and target vector
x = df_data.drop('class', axis=1)
y = df_data['class']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=25)

# Train Decision Tree
model = DecisionTreeClassifier(random_state=25)
model.fit(x_train, y_train)

# Prediction and evaluation
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(model, filled=True, feature_names=x.columns, class_names=['notckd', 'ckd'])
plt.show()
