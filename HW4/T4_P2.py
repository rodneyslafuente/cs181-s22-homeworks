# CS 181, Spring 2022
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Loading datasets for K-Means and HAC
small_dataset = np.load("data/small_dataset.npy")
large_dataset = np.load("data/large_dataset.npy")

# NOTE: You may need to add more helper functions to these classes
class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        # Randomly select cluster centers
        center_idxs = np.random.choice(len(X), self.K, replace=False)
        self.centers = X[center_idxs]

        # Keep track of previous responsibility vector
        last_resp_mat = None

        while True:
            # Initialize responsibility vector
            resp_mat = np.zeros((X.shape[0], self.K))

            # Assign each data point to its closest cluster center
            for i in range(len(X)):
                closest_dist = np.inf
                closest_k = -1
                for k in range(self.K):
                    dist = np.linalg.norm(X[i] - self.centers[k])
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_k = k
                resp_mat[i][closest_k] = 1

            # Update centers
            clust_sums = np.zeros((self.K, X.shape[1]))
            clust_sizes = np.zeros(self.K)
            for resp_vec, x in zip(resp_mat, X):
                clust_sums[np.argmax(resp_vec)] += x
                clust_sizes[np.argmax(resp_vec)] += 1
            self.centers = np.array([s / n for s, n in zip(clust_sums, clust_sizes)])   

            # Check for convergence
            if np.array_equal(last_resp_mat, resp_mat):
                break

            # Update last responsibility vector for comparison
            last_resp_mat = resp_mat

        self.resp_mat = resp_mat

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.centers


class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage

    def __min_linkage(self, C1, C2):
        return np.min(cdist(C1, C2))

    def __max_linkage(self, C1, C2):
        return np.max(cdist(C1, C2))

    def __centroid_linkage(self, C1, C2):
        return np.linalg.norm(np.average(C2, axis=0) - np.average(C1, axis=0))

    # For simplicity and speed, I only merge clusters until there are n_clusters remaining.
    # This removes the need to store the whole dendrogram (e.g. the cluster assignments after
    # every step). I have altered get_mean_images to no longer require n_clusters for this same
    # reason. This approach is supported by https://edstem.org/us/courses/19561/discussion/1316266
    def fit(self, X, n_clusters):
        linkage = None
        if self.linkage == 'min':
            linkage = self.__min_linkage
        elif self.linkage == 'max':
            linkage = self.__max_linkage
        else:
            linkage = self.__centroid_linkage
        
        # Initialize clusters, which are lists of lists of numpy arrays
        self.clusters = [[x] for x in X]
        num_clusters = len(self.clusters)

        while num_clusters > n_clusters:
            # Keep track of indexes to merge, distance between them
            idx_1 = -1
            idx_2 = -1
            min_dist = np.inf

            # Calculate distances
            for i in range(len(self.clusters)):
                for j in range(len(self.clusters)):
                    if i != j:
                        dist = linkage(self.clusters[i], self.clusters[j])
                        if dist < min_dist:
                            min_dist = dist
                            idx_1 = i
                            idx_2 = j
            
            # Merge clusters
            self.clusters[idx_1] += self.clusters[idx_2]
            del self.clusters[idx_2]

            num_clusters = len(self.clusters)

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self):
        return np.array([np.mean(np.array(cluster), 0) for cluster in self.clusters])


# test_X = np.array([[-3,-3],
#                    [-1,-3],
#                    [ 3, 0],
#                    [-2,-1],
#                    [ 0, 0],
#                    [-1,-2]])
# l = 'max'
# hac = HAC(l)
# hac.fit(test_X, 3)
# print(hac.get_mean_images())

# Plotting code for parts 2 and 3
def make_mean_image_plot(data, standardized=False, title=''):
    # Number of random restarts
    niters = 3
    K = 10
    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K)
        KMeansClassifier.fit(data)
        allmeans[:,i] = KMeansClassifier.get_mean_images()
    fig = plt.figure(figsize=(10,10))
    plt.suptitle('Class mean images across random restarts' + (' (standardized data)' if standardized else ''), fontsize=16)
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1+niters*k+i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if k == 0: plt.title('Iter '+str(i))
            if i == 0: ax.set_ylabel('Class '+str(k), rotation=90)
            plt.imshow(allmeans[k,i].reshape(28,28), cmap='Greys_r')
    plt.savefig(title+'plot.png')

# ~~ Part 2 ~~
# make_mean_image_plot(large_dataset, False, title='part2')

# ~~ Part 3 ~~
# large_dataset_standardized = (large_dataset - np.mean(large_dataset)) / np.std(large_dataset)
# make_mean_image_plot(large_dataset_standardized, True, title='part3')

# Plotting code for part 4
LINKAGES = [ 'max', 'min', 'centroid' ]
n_clusters = 10

fig = plt.figure(figsize=(10,10))
plt.suptitle("HAC mean images with max, min, and centroid linkages")
for l_idx, l in enumerate(LINKAGES):
    # Fit HAC
    hac = HAC(l)
    hac.fit(small_dataset, n_clusters)
    mean_images = hac.get_mean_images()
    # Make plot
    for m_idx in range(mean_images.shape[0]):
        m = mean_images[m_idx]
        ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        if m_idx == 0: plt.title(l)
        if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)
        plt.imshow(m.reshape(28,28), cmap='Greys_r')
plt.savefig('part4plot.png')

# TODO: Write plotting code for part 5

# TODO: Write plotting code for part 6