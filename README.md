# CardioV-Disease
Cardiovascular diseases are the leading cause of death globally. It is therefore necessary to identify the causes and develop a system to predict heart attacks in an effective manner. The data below has the information about the factors that might have an impact on cardiovascular health. 

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy import stats
# Load the dataset
df = pd.read_excel('Healthcare.xlsx')

# Drop duplicates
df.drop_duplicates(inplace=True)

# Treat missing values
df.fillna(df.median(), inplace=True)

# Preliminary statistical summary
print(df.describe())

# Plot count plots for categorical variables
cat_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
for var in cat_vars:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=var, data=df)
    plt.title(var)
    plt.show()


# Study occurrence of CVD across age category
age_bins = [18, 30, 40, 50, 60, 70, 80, 90]
df['AgeBin'] = pd.cut(df['age'], age_bins)
plt.figure(figsize=(10, 6))
sns.countplot(x='AgeBin', hue='target', data=df)
plt.title('Occurrence of CVD across age category')
plt.show()

# Study composition of patients with respect to sex category
plt.figure(figsize=(6, 6))
sns.countplot(x='sex', data=df)
plt.title('Composition of patients with respect to sex category')
plt.show()

# Study if heart attacks can be detected based on anomalies in trestbps
plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='trestbps', data=df)
plt.title('Resting blood pressure and occurrence of CVD')
plt.show()

# Describe relationship between cholesterol levels and target variable
plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='chol', data=df)
plt.title('Serum cholesterol and occurrence of CVD')
plt.show()

# Describe relationship between peak exercising and occurrence of heart attack
plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='oldpeak', data=df)
plt.title('ST depression and occurrence of CVD')
plt.show()

# Check if thalassemia is a major cause of CVD
plt.figure(figsize=(10, 6))
sns.countplot(x='thal', hue='target', data=df)
plt.title('Thalassemia and occurrence of CVD')
plt.show()

# Check the correlation between variables
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between variables')
plt.show()

# Pair plot for relationship between variables
sns.pairplot(df, hue='target')
plt.show()

# Split dataset into train and test sets
X = df.drop(['target', 'AgeBin'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('Logistic Regression')
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Train logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Train random forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Evaluate logistic regression model
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

# Evaluate random forest model
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

