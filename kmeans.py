'''kmeans.py
Performs K-Means clustering
Collins Kibet
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors
import random


class KMeans():
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None

        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None

        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data

        # num_samps: int. Number of samples in the dataset
        self.num_samps = None

        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None

        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        self.data = data


        if data is not None:

            self.num_samps, self.num_features = data.shape


    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        data = self.data.copy()

        return data


    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''

        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''


        #subtract pt_1 - pt_2 and sum all values and square root
        euclid_dist = np.sqrt(np.sum(np.square(pt_1 - pt_2)))

        return euclid_dist

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)

        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''

        #initialize self.centroids
        self.centroids = centroids

        #subtract pt_1 - pt_2 and sum all values and square root
        centr_euclid_dist = np.sqrt(np.sum(np.square(pt - centroids), axis = 1))

        return centr_euclid_dist

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''

        #randomly select k numbers from list of self.num_samps and put them in a list

        rand = np.random.randint(self.num_samps, size = k)

        #set initial centroids to K unique randomly selected data samples
        centroids = self.data[rand]



        return centroids

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the difference between all the centroid values from the
        previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the difference
        between every previous and current centroid value is < `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''

        iters = 0
        diff = 100

        #set new_centriods
        new_centroids = self.initialize(k)

        #difference btween new


        while (iters < max_iter and not np.all(np.abs(diff) < tol)):
            iters += 1

            labels = self.update_labels(new_centroids)

            new_centroids, diff = self.update_centroids(k, labels, new_centroids)

        if verbose == True:
            print(iters)
          
        #Initialize k-means variables

        # k: int. Number of clusters
        self.k = k

        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = new_centroids

        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = labels

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = self.compute_inertia()

        return self.inertia


    def cluster_batch(self, k=2, n_iter=1, verbose=False):

        '''
        Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.

        '''

        val = 1000000

        for i in range(n_iter):

            inertia = self.cluster(k)

            if inertia < val:
                val = inertia
                centroids = self.centroids 
                labels = self.data_centroid_labels

        # #debug information
        if verbose == True:
            print(n_iter)

        self.inertia = val
        self.centroids = centroids
        self.data_centroid_labels = labels

        

    def update_labels(self, centroids):

        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''


        dist_to_centroids = []

        #Distance from each point to every centroid
        for i in range(0,self.data.shape[0]):
            x = self.dist_pt_to_centroids(self.data[i], centroids)         
            dist_to_centroids.append(x.tolist())

        dist_to_centroids = np.array(dist_to_centroids)

        labels = []

        #label of each point based on its centroid
        for i in dist_to_centroids:
            l = np.where(i == np.amin(i))[0]
            labels.append(l)

        labels = np.array(labels).flatten()


        return labels


    def update_centroids(self, k, data_centroid_labels, prev_centroids):

        '''
        Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters

        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample

        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------

        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        '''
    
        #python list to add mean of each cluster
        mean_cluster = np.zeros((k,self.num_features))



        #loop through each clusters to get data in each cluster
        for i in range(0, k):
            # determine which points are in that cluster.
            clusters = self.data[data_centroid_labels == i]

            if clusters.size == 0:
                clusters = self.data[np.random.randint(0,self.num_samps)]

            # find the mean of their coordinates
            mean_cluster[i,:] = np.mean(clusters, axis=0)


        #assign mean of each cluster as new centroids
        new_centroids = np.array(mean_cluster)

        #print(self.centroids.shape)

        #difference btwn current and previous values
        cluster_diff = new_centroids - prev_centroids

        return new_centroids, cluster_diff


    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Parameters:
        -----------
        None

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        
        diff = 0

        #for each cluster calculate distance of point to centroid

        for i in range(self.num_samps):
            centroid = self.centroids[self.data_centroid_labels[i]]
            dist = self.dist_pt_to_pt(self.data[i], centroid)
            diff += np.square(dist)

        inertia = diff/self.num_samps

        return inertia


    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        #plot samples belonging to a cluster with same color
        # print(self.centroids)

        for i in range(0,self.k):
            #cluster data
            cls_data = self.data[self.data_centroid_labels == i]
            plt.scatter(cls_data[:,0], cls_data[:,1])
            plt.plot(self.centroids[:,0], self.centroids[:,1], 'k*')




    def elbow_plot(self, max_k):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''

        inertia = []

        for i in range(1,max_k+1):
            inertia.append(self.cluster(i))

        dist = range(1,max_k+1)
        plt.figure(figsize = (8,6))
        plt.plot(dist, inertia, 'bx-', linestyle = '-')
        plt.xlabel('k clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow plot showing the optimal k')


    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''

        data = self.get_data()

        for i in range(self.num_samps):
            indx = self.data_centroid_labels[i]
            data[i,:] = self.centroids[indx.astype(int),:]

        self.data = data

        # #return compressed image
        # data = np.clip(data.astype('uint8'), 0, 255)



