import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Import data into Pandas DataFrame
geo_data = pd.read_csv('mostTraversedLatLong.csv', usecols = ['Lat', 'Long'])
geo_data.head()

# Set up K-Means algorithm
kmeans = KMeans(n_clusters = 3, random_state = 0, n_init="auto")
kmeans.fit(geo_data)
geo_data['Cluster'] = kmeans.labels_

plt.axis("off")
plt.gcf().canvas.manager.set_window_title('mostTraversedLatLong K-Means Clusters')

plt.scatter(geo_data['Long'], geo_data['Lat'], c=geo_data['Cluster'], cmap='rainbow')


# Display the plot

plt.show()

# simple KML package
# https://pypi.org/project/simplekml/
# https://simplekml.readthedocs.io/en/latest/index.html
