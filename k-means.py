import sklearn as sk
import numpy as np
import random
from copy import deepcopy
from matplotlib import pyplot as plt

def generate_random_data(scale = 10, num_datapoints = 100):
    return np.random.randn(num_datapoints, 2).astype('f')*scale

def sample_centroids_from_dataset(dataset, k):
    return dataset[np.random.choice(dataset.shape[0], size = k, replace = False), :]

def plot_graph(data, centroids):
	## Visualising the Data
	plt.scatter(data[:,0], data[:,1])
	plt.scatter(centroids[:,0], centroids[:,1], marker='*', c='r', s=150)

## Function that returns a clusters array, that is the id of the centroid that each point belongs to ##
## and also returns the coordinates of centroids at the end of iteration (when mean turns zero) ##
def run_k_means(data, centroids, k):
    
    centroids_prev = np.zeros(centroids.shape) 
    centroids_current = deepcopy(centroids)
    n = data.shape[0]
    clusters = np.zeros(n)
    distances = np.zeros((n, k))

    error = np.linalg.norm(centroids_current - centroids_prev)

    while error != 0:
        for i in range(k):
            distances[:, i] = np.linalg.norm(data - centroids_current[i], axis = 1)

        clusters = np.argmin(distances, axis = 1)
        centroids_prev = deepcopy(centroids_current)
        
        for i in range(k):
            centroids_current[i] = np.mean(data[clusters == i], axis=0)
        
        error = np.linalg.norm(centroids_current - centroids_prev)
    
    return clusters, centroids_current

np.random.seed(3)
data = generate_random_data()
k = 7

centroids = sample_centroids_from_dataset(data, k)

plot_graph(data, centroids)

clusters, new_centroids = run_k_means(data, centroids, k)

plot_graph(data, new_centroids)