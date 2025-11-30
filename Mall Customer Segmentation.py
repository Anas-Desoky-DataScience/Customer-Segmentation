import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
df = pd.read_csv('Mall_Customers.csv')
df.info()
df.describe()


# Drop CustomerID
df1 = df.drop(['CustomerID','Gender'] , axis=1)

# Histograms
df1.hist(figsize=(12,6))
plt.suptitle("Feature Distributions", fontsize=14)
plt.show()

# Correlation heatmap
sns.heatmap(df1.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

# Standardize features
scaler = StandardScaler()
df1_scaled = scaler.fit_transform(df1)

# Elbow method
inertia = []
K = range(1, 11)
for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df1_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,6))
plt.plot(K, inertia, '-o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# Fit final model
optimal_k = 5
k_means = KMeans(n_clusters=optimal_k, random_state=42)
k_means.fit(df1_scaled)
labels = k_means.labels_

# Add labels to dataset
df1['Cluster'] = labels

# Cluster visualization
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df1['Cluster'],
    palette='rainbow'
)
plt.title('Customer Segmentation by K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
