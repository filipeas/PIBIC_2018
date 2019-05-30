# 
# author:
# Pablo Vinicius - https://github.com/pabloVinicius
#
# contributors:
# Filipe A. Sampaio - https://github.com/filipeas

# imports necessários
import numpy as np
from scipy.spatial.distance import euclidean as ed
import math as mt

def SFc_means(attr_vector, k, seeds, seeds_classes, centroids, threshold, z, total_grouping=0):
    """
    A function, k-means like, that generate k groups in the data entry based
     on pre-estabilished centroids and a precision threshold.

    Args:
        attr_vector: atribute vector of each element (ndarray [MxN]; M is the number of attributes and N the number of elements);
        k: quantity of groups (int);
        seeds: rodulated elements used as first centroids (ndarray [MxZ], M is the number of attributes and Z the number of seeds);
        seeds_classes: array with atributes vector classes; (ndarray [N]; N is the number of elements)
        centroids: position of seeds (ndarray; [Z], Z is the number of seeds, each element represents a position at attr_vector);
        threshold: percetage used to calculate the rotulation precision;
        z: nebulosity level;
        total_grouping: used to set the grouping type (int; 0 to parcial grouping (based on threshold) and 1 to total grouping);
    
    Returns:
        An [N] array of labels, that N is the number of entries.
    """

    def verify_equality(vector, threshold):
        """
        Verify if the first threshold number of vector elemets have the same value.

        Args:
            vector: a list with numbers (list);
            threshold: the number of vector elements to verify (int).
        
        Returns:
            True if the first threshold number of vector elements have the same value and False if they have not.
        """
        actual_value = vector[0]
        for i in range(1,threshold):
            if vector[i] != actual_value:
                return False
            actual_value = vector[i]
        return True


    # generating unique array to avoid repeated data and the positions of the original array elements in the new unique elements array
    histogram, positions_histogram = np.unique(attr_vector, axis=0, return_inverse=True)

    # creating an zeros array to storage the future classes (one for each element from attr_vector)
    labels = np.zeros((len(histogram)))   

    hist_centroids = np.zeros(np.unique(seeds, axis=0).shape[0])
    hist_seeds = np.zeros(np.unique(seeds, axis=0).shape)
    hist_seeds_classes = np.zeros(np.unique(seeds, axis=0).shape[0])
    counter = 0

    for i in range(len(centroids)):
        histogram_position = positions_histogram[centroids[i]]
        labels[histogram_position] = seeds_classes[i]
        if len(np.nonzero(hist_centroids == positions_histogram)[0]) == 0:
            hist_centroids[counter] = histogram_position
            hist_seeds[counter,:] = seeds[i,:]
            hist_seeds_classes[counter] = seeds_classes[i]
            counter += 1
    expoent = 1/(z-1)

    #for each element from attr_vector
    for i in range(len(labels)):
        if len(np.nonzero(hist_centroids == i)[0]) == 0: #but only the not labeled ones
            pertinency_levels = []
            distances = []

            #calculate the euclidean distances between the element and all the centroids
            for j in range(len(hist_centroids)):
                distances.append(ed(histogram[i], hist_seeds[j]))
            
            #calculate the pertinency level of the element to every centroid
            for j in range(len(hist_centroids)):
                mu = 0
                
                for k in range(len(hist_centroids)):
                    mu += (distances[j]/distances[k])**expoent
                
                mu = mu**-1

                pertinency_levels.append([mu, hist_seeds_classes[j]])

            
            #sorting pertinency_levels list in descending order
            pertinency_levels.sort(key=lambda x: x[0], reverse=True)            
            
            pertinency_levels = np.asarray(pertinency_levels)

            if total_grouping == 0:
                #calculating absolute threshold based on percetage threshold and the provided centroids number
                absolute_threshold = mt.floor(len(centroids) * threshold)

                #verify if the first (absolute_threshold) items are from the same group
                #[item[0] for item in pertinency_levels]
                if verify_equality(pertinency_levels[:,1], absolute_threshold):
                    #if yes, the unlabeled data label are the same from the highest pertinency level
                    labels[i] = pertinency_levels[0][1]
            else:
                labels[i] = pertinency_levels[0][1]
    
    final_labels = np.zeros(len(attr_vector))
    for i in range(len(final_labels)):
        final_labels[i] = labels[positions_histogram[i]]

    return final_labels