import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r'C:\Users\henry\OneDrive\Documents\Python Scripts\medical_raw_data.csv')

# Convert Zip codes to string and pad with leading zeros
df['Zip'] = df['Zip'].astype(str).str.zfill(5)

# Timezone Standardization using a mapping dictionary
timezone_mapping = {
    'America/New_York': 'us_eastern', 'America/Detroit': 'us_eastern',
    'America/Chicago': 'us_central', 'America/Denver': 'us_mountain',
    'America/Phoenix': 'us_mountain', 'America/Los_Angeles': 'us_pacific',
    'America/Anchorage': 'us_alaska', 'Pacific/Honolulu': 'us_hawaii',
    'America/Indiana/Indianapolis': 'us_eastern', 'America/Boise': 'us_mountain',
    'America/Kentucky/Louisville': 'us_eastern', 'America/Indiana/Vincennes': 'us_eastern',
    'America/Indiana/Winamac': 'us_eastern', 'America/Indiana/Marengo': 'us_eastern',
    'America/Indiana/Petersburg': 'us_eastern', 'America/Indiana/Vevay': 'us_eastern',
    'America/North_Dakota/Center': 'us_central', 'America/North_Dakota/New_Salem': 'us_central',
    'America/North_Dakota/Beulah': 'us_central', 'America/Menominee': 'us_central',
    'America/Adak': 'us_hawaii', 'Pacific/Guam': 'us_pacific', 'Pacific/Saipan': 'us_pacific',
    'America/Puerto_Rico': 'us_atlantic'
}

df['Timezone'] = df['Timezone'].map(timezone_mapping).astype('category')

# Handle missing values
# For numerical columns: impute with median
numerical_cols = ['Children', 'Age', 'Income', 'Initial_days', 'VitD_levels']
for col in numerical_cols:
    median = df[col].median()
    df[col] = df[col].fillna(median)  # Direct assignment

# For categorical columns: impute with mode
categorical_cols = ['Education', 'Marital', 'Gender', 'Soft_drink', 'Initial_admin']
for col in categorical_cols:
    mode = df[col].mode()[0]
    df[col] = df[col].fillna(mode)  # Direct assignment

# Convert binary fields to 'Yes' and 'No'
binary_cols = ['ReAdmis', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis', 'Diabetes',
               'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
               'Reflux_esophagitis', 'Asthma', 'Soft_drink']
for col in binary_cols:
    if col == 'Anxiety' or col == 'Overweight':
        df[col] = df[col].apply(lambda x: 'Yes' if x == 1.0 else ('No' if x == 0.0 else 'No'))
    else:
        df[col] = df[col].apply(lambda x: 'Yes' if x == 'Yes' else 'No')

# Convert survey responses to ordered categorical
survey_responses = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8']
for response in survey_responses:
    df[response] = pd.Categorical(df[response], categories=[1, 2, 3, 4, 5, 6, 7, 8], ordered=True)

# Rename columns to lowercase with underscores
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# Save the cleaned dataset
df.to_csv(r'C:\Users\henry\OneDrive\Documents\Python Scripts\medical_cleaned_data.csv', index=False)
