import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
df = pd.read_csv(r'C:\Users\henry\OneDrive\Documents\Python Scripts\medical_raw_data.csv', index_col=0)

# Set options for display
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 10)

# Display dataframe information
print(df.info())

# Check uniqueness for key identifiers
def check_uniqueness():
    identifiers = ['CaseOrder', 'Customer_id', 'Interaction', 'UID']
    for identifier in identifiers:
        unique_count = df[identifier].nunique()
        print(f"Unique count in {identifier}: {unique_count} (Expected: {len(df)})")

check_uniqueness()

# Descriptive statistics and outlier checks for numeric columns
def describe_and_outliers(column):
    print(f"\n{column} statistics:")
    print(df[column].describe())
    
    # Outliers using the IQR method
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Number of outliers in {column}: {len(outliers)}")

# Check each numeric column
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_columns:
    describe_and_outliers(col)

# Categorical data checks
def categorical_checks(column):
    print(f"\n{column} value counts:")
    print(df[column].value_counts())

# Check each categorical column
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_columns:
    categorical_checks(col)

# Visualizing missing data with a heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Visualization of Missing Data Across Variables')
plt.show()

# Outlier detection with box plots and histograms for quantitative variables
quantitative_cols = ['TotalCharge', 'Lat', 'Lng', 'Population', 'Income', 'VitD_levels']
for col in quantitative_cols:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot - {col}')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram - {col}')
    plt.show()