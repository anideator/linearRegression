from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

# Load the dataset
df = pd.read_csv("training_data.csv")

# Splitting the data into features (X) and target variable (y)
X = df[['experience', 'education_qualification', 'position']]
y = df['salary']

# One-hot encode the categorical features
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [1, 2])], remainder='passthrough')
X_encoded = ct.fit_transform(X)

# Get feature names after one-hot encoding
feature_names_encoded = ct.named_transformers_['encoder'].get_feature_names_out()

# Concatenate the feature names after one-hot encoding with remaining features
all_feature_names = list(feature_names_encoded) + ['experience']

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initializing and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating the Mean Squared Error (MSE) to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

yearsOfExp = int(input("Years of Experience: "))
eduQualification = input("Educational Qualification (Bachelor / High School / Masters / PhD): ")
jobPosition = input("Job Position (Junior / Senior / Manager): ")
# Predicting salary for a sample instance
sample_instance = [[yearsOfExp, eduQualification, jobPosition]]  # Example: 5 years of experience, Master's degree, Junior position
sample_instance_encoded = ct.transform(sample_instance)
predicted_salary = model.predict(sample_instance_encoded)[0]
print("Predicted salary for sample instance:", predicted_salary)
