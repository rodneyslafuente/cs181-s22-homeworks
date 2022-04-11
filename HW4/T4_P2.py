# CS 181, Spring 2022
# Homework 4

from cgitb import small
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

import torchvision
import numpy as np

mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)  # download MNIST
N = 6000 

x = mnist_trainset.data[:N]  # select N datapoints
x = x.flatten(1)             # flatten the images
x = x.float()                # convert pixels from uint8 to float
x = x.numpy()  

# Loading datasets for K-Means and HAC
# small_dataset = np.load("data/small_dataset.npy")
# large_dataset = np.load("data/large_dataset.npy")

# NOTE: You may need to add more helper functions to these classes
class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X, plot_rss=False):
        # Randomly select cluster centers
        center_idxs = np.random.choice(len(X), self.K, replace=False)
        self.centers = X[center_idxs]

        # Keep track of previous responsibility vector
        last_resp_mat = None

        losses = []
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

            # Calculate residual sum of squares 
            if plot_rss:
                loss = 0
                for i in range(len(X)):
                    for k in range(self.K):
                        if resp_mat[i][k]:
                            loss += np.linalg.norm(X[i] - self.centers[k]) ** 2
                losses.append(loss)

            # Check for convergence
            if np.array_equal(last_resp_mat, resp_mat):
                break

            # Update last responsibility vector
            last_resp_mat = resp_mat

        self.resp_mat = resp_mat
        self.losses = np.array(losses)

        if plot_rss:
            _, ax = plt.subplots()
            ax.plot(range(len(losses)), losses)
            ax.set_ylabel("RSS")
            ax.set_xlabel("Iterations")
            plt.savefig("part1plot.png")

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


# ~~ Part 1 ~~
KMeansClassifier = KMeans(10)
KMeansClassifier.fit(x, plot_rss=True)
rss = KMeansClassifier.losses[-1]
print('RSS: '+str(rss)+' Mean RSS: '+str(rss / x.shape[0]))


# ~~ Part 2 ~~
# make_mean_image_plot(large_dataset, False, title='part2')

# ~~ Part 3 ~~
# large_dataset_standardized = (large_dataset - np.mean(large_dataset)) / np.std(large_dataset)
# make_mean_image_plot(large_dataset_standardized, True, title='part3')

# Plotting code for part 4
# LINKAGES = [ 'max', 'min', 'centroid' ]
# n_clusters = 10

# fig = plt.figure(figsize=(10,10))
# plt.suptitle("HAC mean images with max, min, and centroid linkages")
# for l_idx, l in enumerate(LINKAGES):
#     # Fit HAC
#     hac = HAC(l)
#     hac.fit(small_dataset, n_clusters)
#     mean_images = hac.get_mean_images()
#     # Make plot
#     for m_idx in range(mean_images.shape[0]):
#         m = mean_images[m_idx]
#         ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
#         plt.setp(ax.get_xticklabels(), visible=False)
#         plt.setp(ax.get_yticklabels(), visible=False)
#         ax.tick_params(axis='both', which='both', length=0)
#         if m_idx == 0: plt.title(l)
#         if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)
#         plt.imshow(m.reshape(28,28), cmap='Greys_r')
# plt.savefig('part4plot.png')

# ~~ Part 5 ~~
# fig, ax = plt.subplots()
# ax.set_title('KMeans')
# ax.set_xlabel('Cluster index')
# ax.set_ylabel('Number of images in cluster')
# KMeansClassifier = KMeans(10)
# KMeansClassifier.fit(small_dataset)
# clust_sizes = np.sum(KMeansClassifier.resp_mat, axis=0)
# ax.plot(np.sum(KMeansClassifier.resp_mat, axis=0), 'o')
# plt.savefig('part5kmeansplot.png')

# fig, ax = plt.subplots()
# ax.set_title('HAC, min linkage')
# ax.set_xlabel('Cluster index')
# ax.set_ylabel('Number of images in cluster')
# hac = HAC('min')
# hac.fit(small_dataset, 10)
# clust_sizes = [len(clust) for clust in hac.clusters]
# ax.plot(clust_sizes, 'o')
# plt.savefig('part5hacmin.png')

# fig, ax = plt.subplots()
# ax.set_title('HAC, max linkage')
# ax.set_xlabel('Cluster index')
# ax.set_ylabel('Number of images in cluster')
# hac = HAC('max')
# hac.fit(small_dataset, 10)
# clust_sizes = [len(clust) for clust in hac.clusters]
# ax.plot(clust_sizes, 'o')
# plt.savefig('part5hacmax.png')

# fig, ax = plt.subplots()
# ax.set_title('HAC, centroid linkage')
# ax.set_xlabel('Cluster index')
# ax.set_ylabel('Number of images in cluster')
# hac = HAC('centroid')
# hac.fit(small_dataset, 10)
# clust_sizes = [len(clust) for clust in hac.clusters]
# ax.plot(clust_sizes, 'o')
# plt.savefig('part5haccentroid.png')

# ~~ Part 6 ~~
# from seaborn import heatmap

# uniform_data = np.random.rand(10, 12)

# print('kmeans')
# KMeansClassifier = KMeans(10)
# KMeansClassifier.fit(small_dataset)
# np.save('m1', KMeansClassifier.resp_mat)


# hac = HAC('centroid')
# hac.fit(small_dataset, 10)
# np.save('c3', hac.clusters, allow_pickle=True)

# def get_index(arr, elt):
#     i = 0
#     while not np.array_equal(arr[i], elt):
#         i += 1
#     return i

# resp_mat = np.zeros((len(small_dataset), 10))
# clusters = np.load('c3.npy', allow_pickle=True)
# for k in range(len(clusters)):
#     for x in clusters[k]:
#         idx = get_index(small_dataset, x)
#         resp_mat[idx][k] = 1

# np.save('m4', resp_mat)
# print(resp_mat)
# print(np.sum(resp_mat, axis=0))

# def generate_mat(m1, m2):
#     mat = np.zeros((10, 10))
#     for i in range(len(m1)):
#         for j in range(len(m2)):
#             mat[np.argmax(m1[i])][np.argmax(m2[j])] += 1
#     return mat

# m1 = np.load('m1.npy') # kmeans
# m2 = np.load('m2.npy') # min
# m3 = np.load('m3.npy') # max
# m4 = np.load('m4.npy') # centroid

# data = generate_mat(m3, m4)

# fig, ax = plt.subplots()
# ax = heatmap(data)
# ax.set_ylabel('HAC, max linkage')
# ax.set_xlabel('HAC, centroid linkage')
# plt.savefig('part6maxcentroid.png')