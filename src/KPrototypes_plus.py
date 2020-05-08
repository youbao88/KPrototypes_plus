import pandas as pd
import numpy as np
from numba import jit
from joblib import Parallel, delayed

@jit(nopython = True)
def calculate_dissimilarity(row1, row2, types, gamma):
    '''
    Calculate the dissimilarity between two rwos of data.
    
    Args:
        row1, row2: Rows of data from dataset.
        types: A numpy array that indicates if the variable is categorical or numerical. For example: types = [1,1,0,0,0,0] means the first two variables are categorical and the last four variables are numerical.
        gamma: A value that controls how algorithm favours categorical variables.
    
    Returns:
        The dissimilarity.
    '''
    n = row1.shape[0]
    D = 0.0
    for k in range(n):
        if types[k] == 0:
            D += np.square(row1[k] - row2[k])
        else:
            if row1[k] != row2[k]:
                D += 1.0*gamma
    return D

@jit(nopython = True)
def cost_function(data, cluster_label, prototype, types, gamma):
    '''
    Calculate cost function
    '''
    cost = 0.0
    for i in range(data.shape[0]):
        cost += calculate_dissimilarity(data[i], prototype[cluster_label[i]], types = types, gamma = gamma)
    return cost

@jit(nopython = True)
def hyper_k_prototype(data, n_cluster, types, gamma,initial_prototypes):
    '''
    Implement k_prototype function
    '''
    prototypes = initial_prototypes
    old_cluster_label = np.zeros(data.shape[0], dtype = np.int32)
    n_iters = 0
    while True:
        new_cluster_label = np.zeros(data.shape[0], dtype = np.int32)
        for i in range(data.shape[0]):
            cluster_label_with_minimal_distance = None
            minimal_distance = None
            for j in range(n_cluster):
                current_distance = calculate_dissimilarity(data[i], prototypes[j], types = types, gamma = gamma)
                if cluster_label_with_minimal_distance is None:
                    cluster_label_with_minimal_distance = j
                    minimal_distance = current_distance
                else:
                    if current_distance < minimal_distance:
                        minimal_distance = current_distance
                        cluster_label_with_minimal_distance = j
            new_cluster_label[i] = cluster_label_with_minimal_distance
        
        identical_flag = 1
        for i in range(new_cluster_label.shape[0]):
            if new_cluster_label[i] != old_cluster_label[i]:
                identical_flag = 0
                break
        if identical_flag == 1:
            break
        for i in range(n_cluster):
            for j in range(data.shape[1]):
                if types[j] == 1:
                    counts = np.bincount(data[new_cluster_label == i][:,j].astype(np.int32))
                    prototypes[i,j] = np.argmax(counts)
                else:
                    prototypes[i,j] = np.mean(data[new_cluster_label == i][:,j])
        old_cluster_label = new_cluster_label
        n_iters += 1
    cost = cost_function(data = data, cluster_label = new_cluster_label, prototype = prototypes, types = types, gamma = gamma)
    return (new_cluster_label, prototypes, n_iters, cost)

@jit(nopython = True)
def cao_initialization(cat_data, n_cluster):
    '''
    Implementation of Cao's initialization for categorical variables
    '''
    cat_prototype = np.zeros((n_cluster, cat_data.shape[1]), dtype = np.int32)
    for n in range(n_cluster):
        if n==0:
            max_density = None
            max_i = None
            for i in range(cat_data.shape[0]):
                cur_density = 0
                for j in range(cat_data.shape[1]):
                    cur_density += (np.bincount(cat_data[:,j])[cat_data[i,j]])/cat_data.shape[0]
                cur_density = cur_density/cat_data.shape[1]
                if max_density is None or max_density < cur_density:
                    max_density = cur_density
                    max_i = i
            cat_prototype[int(n)] = cat_data[int(max_i)]
            previous_i = max_i
        else:
            max_product = None
            max_i = None
            for i in range(cat_data.shape[0]):
                cur_density = 0
                cur_distance = 0
                for j in range(cat_data.shape[1]):
                    cur_density += (np.bincount(cat_data[:,j])[cat_data[i,j]])/cat_data.shape[0]
                    if cat_data[int(i),int(j)] != cat_data[int(previous_i),int(j)]:
                        cur_distance += 1
                cur_product = cur_density*cur_distance
                if max_product is None or max_product < cur_product:
                    max_product = cur_product
                    max_i = i
            cat_prototype[int(n)] = cat_data[int(max_i)]
            previous_i = max_i
    return cat_prototype

@jit(nopython = True)
def num_initialization(num_data, n_cluster):
    '''
    Implementation of k-means++ initialization for numeric variables
    '''
    num_prototype = np.zeros((n_cluster, num_data.shape[1]), dtype = np.float64)
    for n in range(n_cluster):
        if n==0:
            num_prototype[int(n)] = num_data[np.random.randint(num_data.shape[0])]
        else:
            D_array = np.zeros(num_data.shape[0])
            sum_all = 0
            for i in range(num_data.shape[0]):
                Dx = None
                for _n in range(0,n):
                    distance = np.sqrt(np.sum(np.square(num_prototype[_n]-num_data[i])))
                    if Dx is None or Dx > distance:
                        Dx = distance
                sum_all += Dx
                D_array[i] = Dx
            sum_all *= np.random.random_sample()
            for i, di in enumerate(D_array):
                sum_all -= di
                if sum_all <= 0:
                    num_prototype[int(n)] = num_data[int(i)]
                    break            
    return num_prototype

@jit(nopython = True)
def concact_init(cao_init, num_init, n_clusters, types):
    '''
    Concact initialization of categorical variables and numeric variables into one
    '''
    init_prototype = np.zeros((n_clusters, cao_init.shape[1]+num_init.shape[1]))
    num_cat = 0
    num_num = 0
    for i in range(len(types)):
        if types[i] == 1:
            init_prototype[:,i] = cao_init[:,num_cat]
            num_cat += 1
        else:
            init_prototype[:,i] = num_init[:,num_num]
            num_num += 1
    return init_prototype

@jit(nopython = True)
def mean_std(data, types):
    std = 0
    count_num_column = 0
    for col_index in range(len(types)):
        if types[col_index] == 0:
            count_num_column += 1
            std += np.std(data[:,col_index])
    return std/count_num_column



class KPrototypes_plus:
    '''A high speed JIT class used to run k prototpye clustering algorithm on both categorical
    and numerical data

    Attributes:
        X: a 2-D numpy array
        n_clusters: the number of clusters
        n_init: the number of parallel oprations by using different initializations
        gamma (optional) : A value that controls how algorithm favours categorical variables.
                            The default is the mean std of all numeric variables
        n_jobs (optional): The number of parallel processors:
                            The default is -1, which means useing all the processor
        types: A numpy array that indicates if the variable is categorical or numerical. 
                For example: types = [1,1,0,0,0,0] means the first two variables are categorical 
                and the last four variables are numerical.
        

        labels_: Cluster labels
        cluster_centroids_: Centroids (prototypes) of each cluster
        n_iter_: Number of iterations
        cost_: The costs of the finial model


    
    Methods: predict(): a single iteration
            fit_predict(X, categorical): fit the data into the model 
                                                    and calculate returns
    
    To_use:

        >>> model = KPrototypes_plus(n_clusters = 3, n_init = 4, gamma = None, n_jobs = -1)  #initialize the model
        >>> model.fit_predict(X=df, categorical = [0,1])  #fit the data and categorical into the mdoel

        >>> model.labels_                          #return the cluster_labels
        >>> model.cluster_centroids_               #return the cluster centroid points(prototypes)
        >>> model.n_iter_                          #return the number of iterations
        >>> model.cost_                            #return the costs

    '''

    def __init__(self, n_clusters, n_init, gamma = None, n_jobs=-1):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.gamma = gamma
        self.n_jobs = n_jobs
    
    def predict(self):
        cat_data = self.data[:,self.types == 1].astype(np.int64)
        cat_init = cao_initialization(cat_data, self.n_clusters)
        num_data = self.data[:,self.types == 0]
        num_init = num_initialization(num_data, self.n_clusters)
        initialization = concact_init(cat_init, num_init, n_clusters = self.n_clusters, types = self.types)
        output = hyper_k_prototype(data = self.data, n_cluster = self.n_clusters, types = self.types, gamma = self.gamma,initial_prototypes = initialization)
        return output

    def fit_predict(self, X, categorical):
        self.data = X.values
        self.types = np.array([1 if x in categorical else 0 for x in range(self.data.shape[1])])
        if self.gamma is None:
            self.gamma = mean_std(self.data, self.types)/2
        outputs = Parallel(n_jobs=self.n_jobs)(delayed(self.predict)() for _ in range(self.n_init))
        costs = [x[3] for x in outputs]
        min_index = np.argmin(np.asarray(costs))
        self.labels_ = outputs[min_index][0]
        self.cluster_centroids_ = outputs[min_index][1]
        self.n_iter_ = outputs[min_index][2]
        self.cost_ = outputs[min_index][3]