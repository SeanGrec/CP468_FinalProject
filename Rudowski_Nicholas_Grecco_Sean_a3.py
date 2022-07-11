#+++++++++++++++++++++++++++++++++++++
# CP 468: Spring Term 2022
# Assignment 3: Question 5
# Grecco, Sean - 170317470
# Rudowski, Nicholas - 191963270
#+++++++++++++++++++++++++++++++++++++
# Notes:
# ->
#+++++++++++++++++++++++++++++++++++++

import math
import matplotlib.pyplot as plt
import pandas as pd

# Global max var used for min_dist
MAX = 99999


# Take in a file name and k (number of clusters)
def kmeans(file, k):
    # Loads in data using pandas
    data = pd.read_csv(file)
    # Plot initial data
    data.plot(x="f1", y="f2", kind="scatter")
    plt.show()
    # Initialize empty centroids array and empty clusters
    centroids = []
    clusters = []
    # Go Through and fills in initial centroids and clusters with first k
    # points
    for i in range(k):
        clusters.append((data.iloc[i], []))
        centroids.append(data.iloc[i])
    # Keep track of iterations
    iterations = 0
    # initialize old centroids
    oldCentroids = None
    # Run the main k-means algorithmn, go until the old centroids and current
    # are the same
    while not isEqual(centroids, oldCentroids, k):
        # Save old centroids
        oldCentroids = centroids
        # increment iterations
        iterations += 1
        # Go through and clears clusters
        for i in range(k):
            clusters[i] = (clusters[i][0], [])
        # For each point in the data go through
        for point in data.iterrows():
            # Initialize first min_dist as max
            min_dist = (MAX, 0)
            # keep track of min index
            min_i = 0
            # Go through the centroids in each clusters
            for centroid in clusters:
                # Calculates euclidian distance from data point to centroid
                dist = euclidianDist(point, centroid)
                # Keep track of closest centroid
                if min_dist[0] >= dist:
                    min_dist = (dist, min_i)
                min_i += 1
            # Store point in closest centroid cluster
            clusters[min_dist[1]][1].append(point)
        # Assign centroids based on datapoint labels
        centroids = []
        for i in range(k):
            # Vars used for average distance
            sum_x = 0
            sum_y = 0
            size = 0
            # Goes through each clusters points to calculate the new centroids
            # distance
            for data_point in clusters[i][1]:
                sum_x += data_point[0]
                sum_y += data_point[1]
                size += 1
            # New centroids x and y are created based off average dist in
            # cluster
            new_centroid = [sum_x / size, sum_y / size]
            # Add new centroids to centroids array
            centroids.append(new_centroid[1])
            # Add new centroid as centroid to the cluster
            clusters[i] = (new_centroid[1], clusters[i][1])
    # Return the centroids and the clusters
    return centroids, clusters


# Calculates euclidian distance of given point to centroid
def euclidianDist(point, centroid):
    return math.sqrt(math.sqrt((point[1][0] - centroid[0][0])
                               ** 2 + (point[1][1] - centroid[0][1])**2))


# checks if the centroids are the same
def isEqual(centroids, oldCentroids, k):
    if(type(centroids) != type(oldCentroids)):
        return False
    for i in range(k):
        if not centroids[i].equals(oldCentroids[i]):
            return False
    return True


#-------------------------------------------


if __name__ == "__main__":
    # Set k to 2
    k = 2
    # Call kmeans function and save return values
    final_centroid, final_clusters = kmeans("kmeans.csv", k)
    # Turn the clusters into dataframes
    c1 = []
    for i in final_clusters[0][1]:
        c1.append([i[1][0], i[1][1]])
    df1 = pd.DataFrame(c1, columns=["x", "y"])
    c2 = []
    for i in final_clusters[1][1]:
        c2.append([i[1][0], i[1][1]])
    df2 = pd.DataFrame(c2, columns=["x", "y"])
    # Plot the dataframes as scatterplots.
    ax = df1.plot(x="x", y="y", kind="scatter", color="green")
    df2.plot(ax=ax, x="x", y="y", kind="scatter", color="red")
    plt.show()
    # Print final outputs
    print("FINAL CLUSTER SIZES: c1: " +
          str(len(c1)) + " AND c2: " + str(len(c2)))

    print("FINAL CENTROIDS: ")
    print("centroid1: " + str(final_centroid[0]))
    print("centroid2: " + str(final_centroid[1]))
