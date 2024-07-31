import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Specify the path to the CSV file
file_path = r'C:\Users\henry\OneDrive\Documents\Python Scripts\medical_cleaned_data.csv'

# Load the data from CSV file
df = pd.read_csv(file_path)

# Select only the quantitative variables needed for PCA
features = ["lat", "lng", "population", "children", "age", "income", "vitd_levels", "doc_visits", "full_meals_eaten", "vitd_supp"]
x = df.loc[:, features].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

# Perform PCA
pca = PCA(n_components=min(11, len(features)))
principalComponents = pca.fit_transform(x)

# Create a DataFrame with the principal components
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])

# Calculate and display the PCA loadings matrix
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'], index=features)
print(loadings)

# Scree plot to visualize the explained variance by each component
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained')
plt.axhline(y=0.1, color='r', linestyle='-')
plt.xticks(range(1, len(features) + 1))
plt.show()
