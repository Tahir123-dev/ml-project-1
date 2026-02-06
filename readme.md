# ML Project 1
Insurance Charges Prediction Project
Project Overview

This project analyzes an insurance dataset to understand the factors that influence medical insurance charges and prepares the data for building a predictive machine learning model.

The main objective is to predict insurance charges by analyzing demographic, lifestyle, and health-related features using data exploration, preprocessing, feature engineering, and statistical feature selection.

Target Variable

charges

Dataset Description

The dataset (insurance.csv) contains 1338 records with 7 original features:

Column	Description
age	Age of the insured person
sex	Gender (male, female)
bmi	Body Mass Index
children	Number of dependents
smoker	Smoking status (yes, no)
region	Residential region
charges	Medical insurance cost (target variable)
Step 1: Data Loading
df = pd.read_csv('insurance.csv')


Loads the dataset into the DataFrame df

Ensures data is accessible for analysis and preprocessing

Step 2: Exploratory Data Analysis (EDA)
Dataset Shape
df.shape


Rows: 1338

Columns: 7

Data Structure
df.info()


Numeric columns: age, bmi, children, charges

Categorical columns: sex, smoker, region

No missing values detected

Statistical Summary
df.describe()


Provides statistical insights such as mean, standard deviation, minimum, and maximum values for numerical features.

Missing Values Check
df.isnull().sum()


Confirms that the dataset contains no null values.

Step 3: Data Visualization

Histograms for numeric columns using numeric_columns

Count plots for categorical features (sex, smoker, children)

Box plots for detecting outliers

Correlation heatmap to analyze relationships between numerical variables

Key observation:

charges shows strong correlation with smoker

Moderate correlation with age and bmi

Step 4: Data Cleaning
Duplicate Removal
df_cleaned = df.copy()
df_cleaned.drop_duplicates(inplace=True)


Removes duplicate rows

Final dataset size: 1337 rows

Step 5: Encoding Categorical Variables
Binary Encoding
df_cleaned['sex'] = df_cleaned['sex'].map({"male": 0, "female": 1})
df_cleaned['smoker'] = df_cleaned['smoker'].map({"no": 0, "yes": 1})

Renaming Columns for Clarity
df_cleaned.rename(columns={
    'sex': 'is_female',
    'smoker': 'is_smoker'
}, inplace=True)

Step 6: One-Hot Encoding of Region
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)


Creates:

region_northwest

region_southeast

region_southwest

The northeast region is treated as the reference category.

Step 7: Feature Engineering (BMI Categories)
Creating BMI Categories
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

Encoding BMI Categories
df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)


Generated features:

bmi_category_Normal

bmi_category_Overweight

bmi_category_Obese

Step 8: Feature Scaling
cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])


Scaling ensures:

All numeric features are on a similar scale

Better performance for machine learning models

Step 9: Pearson Correlation Analysis
pearsonr(df_cleaned[feature], df_cleaned['charges'])


Top correlated features with charges:

Feature	Pearson Correlation
is_smoker	0.787
age	0.298
bmi	0.196
bmi_category_Obese	0.200

Smoking status is the most influential factor affecting insurance charges.

Step 10: Chi-Square Test for Categorical Features
Charges Binning
df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)

Chi-Square Test
chi2_contingency(contingency)

Features Retained (p < 0.05)

is_smoker

region_southeast

is_female

bmi_category_Obese

Features Dropped

region_northwest

region_southwest

bmi_category_Normal

bmi_category_Overweight

Step 11: Final Feature Set
final_df = df_cleaned[
    ['age', 'is_female', 'bmi', 'children', 'is_smoker',
     'charges', 'region_southeast', 'bmi_category_Obese']
]

Final Dataset Columns
Column	Description
age	Standardized age
is_female	Gender indicator
bmi	Standardized BMI
children	Number of dependents
is_smoker	Smoking indicator
region_southeast	Regional indicator
bmi_category_Obese	Obesity indicator
charges	Target variable

## Project Structure
- data/
- notebooks/
- src/
- requirements.txt

## How to Run
1. Install dependencies
2. Run the notebook or script

## Author
Tahir Fareed
