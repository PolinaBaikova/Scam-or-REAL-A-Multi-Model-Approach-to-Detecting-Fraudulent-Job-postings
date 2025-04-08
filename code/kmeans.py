# Load the necessary libraries
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

# Establish a connection to the SQLite database (or create it if it doesn't exist)
con = sqlite3.connect("RealFakeJobs.db")
# Create a cursor object to interact with the database
cur = con.cursor()


# Select columns from Job_Posts_US
cur.execute('''
    SELECT 
        Employment_type,
        Required_experience,
        Required_education,
        Fraudulent
    FROM Job_Posts_US;
''')

# Load the results into a pandas DataFrame
df = pd.DataFrame(cur.fetchall(), columns=[
    'Employment_type', 'Required_experience', 'Required_education', 'Fraudulent'
])



# Define ordinal mappings for categorical columns

# Employment_type order
employment_order = {'unspecified': 0, 'other': 0, 'temporary': 1, 'part-time': 2, 'contract': 3, 'full-time': 4}

# Required_experience order
experience_order = {'unspecified': 0, 'not applicable': 0, 'internship': 1, 'entry level': 2, 'associate': 3,
    'mid-senior level': 4, 'director': 5, 'executive': 6}

# Required_education order
education_order = {'unspecified': 0, 'some high school': 1, 'high school or equivalent': 2, 'some college': 3,
    'vocational': 4, 'associate degree': 5, 'certification': 6, 'bachelor\'s degree': 7, 'professional': 8,
    'master\'s degree': 9, 'doctorate': 10}

# Apply the mappings
df['Employment_type'] = df['Employment_type'].map(employment_order)
df['Required_experience'] = df['Required_experience'].map(experience_order)
df['Required_education'] = df['Required_education'].map(education_order)


# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


# Finding the optimal number of clusters
inertia = []
silhouette_scores = []
cluster_range = range(2, 10)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))


# Plot Elbow Method & Silhouette Score
plt.figure(figsize=(9, 3))

plt.subplot(1, 2, 1)
plt.plot(cluster_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')

plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='s', color='red')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal K')

plt.show()


# Run K-Means 
optimal_k = 7
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
df['cluster'] = kmeans.fit_predict(scaled_data)


# Reduce to 3D with PCA
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(scaled_data)

# Add PCA components to the DataFrame
df['pca1'] = pca_result_3d[:, 0]
df['pca2'] = pca_result_3d[:, 1]
df['pca3'] = pca_result_3d[:, 2]

# Transform cluster centers into PCA space
centers_pca_3d = pca_3d.transform(kmeans.cluster_centers_)



# Create 3D Scatter Plot
fig = px.scatter_3d(
    df,
    x='pca1', y='pca2', z='pca3',
    color=df['cluster'].astype(str),  # Convert to string for legend
    title='K-Means Clustering of Job Postings',
    opacity=0.7
)

# Add Cluster Centers 
fig.add_trace(go.Scatter3d(
    x=centers_pca_3d[:, 0],
    y=centers_pca_3d[:, 1],
    z=centers_pca_3d[:, 2],
    mode='markers',
    marker=dict(size=4, color='black', symbol='x', opacity=1),
    showlegend=False
))
fig.update_traces(marker=dict(size=4), selector=dict(mode='markers')) # Adjust points size
fig.update_layout(
    legend_title_text='Cluster')
fig.show()



# Convert cluster centers from scaled values back to original feature values
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
# Create a DataFrame to store the cluster centers with meaningful column names
centroids_df = pd.DataFrame(
    centroids_original,
    columns=['Employment_type', 'Required_experience', 'Required_education', 'Fraudulent']
)
print(centroids_df.round(2))



