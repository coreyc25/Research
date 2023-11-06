import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import simplekml

# Import data into Pandas DataFrame
geo_data = pd.read_csv('mostTraversedLatLong.csv', usecols=['Lat', 'Long'])

# Set up K-Means algorithm
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
kmeans.fit(geo_data)
geo_data['Cluster'] = kmeans.labels_

# Define a list of colors for clusters
cluster_colors = ['red', 'green', 'blue']  # You can customize the colors

# Create a KML object
kml = simplekml.Kml()

# Loop through each cluster and add points to KML with cluster-specific colors
for cluster_id in geo_data['Cluster'].unique():
    cluster_points = geo_data[geo_data['Cluster'] == cluster_id]
    for index, row in cluster_points.iterrows():
        pnt = kml.newpoint(
            name=f'Cluster {cluster_id}',
            coords=[(row['Long'], row['Lat'])],
            )
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/ms/icons/' + cluster_colors[cluster_id] + '-dot.png'

# Save the KML file
kml.save('clusters.kml')

# Create a scatter plot (optional)
plt.axis("off")
plt.gcf().canvas.manager.set_window_title('mostTraversedLatLong K-Means Clusters')
plt.scatter(geo_data['Long'], geo_data['Lat'], c=geo_data['Cluster'], cmap='rainbow')
plt.show()
