import pandas as pd
import numpy as np

# Define the number of samples in the dataset
num_samples = 1000

# Generate random values for features
experience = np.random.randint(0, 20, num_samples)  # Assuming experience is in years
education_qualification = np.random.choice(['High School', 'Bachelor', 'Masters', 'PhD'], num_samples)
position = np.random.choice(['Junior', 'Senior', 'Manager'], num_samples)

# Define a function to calculate salary based on the features
def calculate_salary(experience, education_qualification, position):
    base_salary = 30000  # Base salary
    education_bonus = {'High School': 0, 'Bachelor': 5000, 'Masters': 10000, 'PhD': 15000}  # Bonus based on education
    position_bonus = {'Junior': 0, 'Senior': 10000, 'Manager': 20000}  # Bonus based on position
    
    total_bonus = education_bonus[education_qualification] + position_bonus[position]
    total_salary = base_salary + (experience * 2000) + total_bonus  # Assuming $2000 increase per year of experience
    return total_salary

# Calculate salary for each sample
salary = [calculate_salary(exp, edu, pos) for exp, edu, pos in zip(experience, education_qualification, position)]

# Create a DataFrame to store the dataset
data = pd.DataFrame({
    'experience': experience,
    'education_qualification': education_qualification,
    'position': position,
    'salary': salary
})

# Save the dataset to a CSV file
data.to_csv('training_data.csv', index=False)

print("Training data saved successfully!")
