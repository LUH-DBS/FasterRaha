

import json
import math
import multiprocessing as mp
import os

import fastcluster
import hdbscan
import kmodes.kmodes as km
import numpy
import pandas as pd
import parc
import raha as rh
import scipy
import sklearn as sl
from joblib import Parallel, delayed

#Implements faster cluster algorithms for raha through changes in the build_cluster function

class FasterDetection(rh.detection.Detection):

    

    def __init__(self):
        super().__init__()

        self.algorithms = ['kmeans', 'mbatch', 'hdbscan', 'agglomerative_average', 'kmodes', 'parc', 'birch', 'agglomerative_single']
        self.FEATUREREDUCTION = False
        self.N_JOBS = 1
        self.CLUSTER_ALGORITHM = 'kmeans' # ['kmeans', 'mbatch', 'hdbscan', 'agglomerative', 'kmodes', 'parc', 'birch', 'agglomerative_single'] # ['kmeans', 'agglomerative_single', 'agglomerative'] are recommended. All other algoriths have parameters that need to be adjusted.

        ####configurations if needed####
        self.MBATCH_SIZE = 0.6
        self.BIRCH_THRESH = -1
        self.HDBSCAN_MIN_CLUSTER_SIZE = 2
        self.HDBSCAN_MIN_SAMPLES = 100
        self.PARC_DIST_STD_LOCAL = 4
        self.PARC_JAC_STD_GLOBAL = 0.15
    
    
    def generate_features(self, d):
        """
        This method generates features.
        """
        d.NAME_INDEX = {}
        columns_features_list = []
        for j in range(d.dataframe.shape[1]):
            
            feature_vectors = numpy.zeros((d.dataframe.shape[0], len(d.strategy_profiles)))
            for strategy_index, strategy_profile in enumerate(d.strategy_profiles):
                strategy_name = json.loads(strategy_profile["name"])[0]
                if strategy_name in self.ERROR_DETECTION_ALGORITHMS:
                    for cell in strategy_profile["output"]:
                        if cell[1] == j:
                            feature_vectors[cell[0], strategy_index] = 1.0
                    ###### store strategy names for later #####
                    try:
                        d.NAME_INDEX[j].append(strategy_name)
                    except:
                        d.NAME_INDEX[j] = [strategy_name]
                    ###########################################
            if "TFIDF" in self.ERROR_DETECTION_ALGORITHMS:
                vectorizer = sl.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words="english")
                corpus = d.dataframe.iloc[:, j]
                try:
                    tfidf_features = vectorizer.fit_transform(corpus)
                    feature_vectors = numpy.column_stack((feature_vectors, numpy.array(tfidf_features.todense())))
                except:
                    pass
            non_identical_columns = numpy.any(feature_vectors != feature_vectors[0, :], axis=0)
            
            ##### filter unused strategy names #####
            li = []
            for name,non_identical in zip(d.NAME_INDEX[j], non_identical_columns):
                if non_identical:
                    li.append(name)
            
            d.NAME_INDEX[j] = li
            #######################################

            feature_vectors = feature_vectors[:, non_identical_columns]
            if self.VERBOSE:
                print("{} Features are generated for column {}.".format(feature_vectors.shape[1], j))
            columns_features_list.append(feature_vectors)
        d.column_features = columns_features_list

    def build_cluster_agglomerative(self, d, j):
        """
        This method builds clusters using agglomerative clustering.
        """

        feature_vectors = d.column_features[j]
        clusters_k_c_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        cells_clusters_k_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        try:
            clustering_model = scipy.cluster.hierarchy.linkage(feature_vectors, method="average", metric="cosine")
            for k in clusters_k_c_ce:
                model_labels = [l - 1 for l in
                                scipy.cluster.hierarchy.fcluster(clustering_model, k, criterion="maxclust")]
                for index, c in enumerate(model_labels):
                    if c not in clusters_k_c_ce[k]:
                        clusters_k_c_ce[k][c] = {}
                    cell = (index, j)
                    clusters_k_c_ce[k][c][cell] = 1
                    cells_clusters_k_ce[k][cell] = c
        except Exception as e:
            print(e)
            pass
        if self.VERBOSE:
            print("A hierarchical clustering model with average-linkage is built for column {}.".format(j))
        return [clusters_k_c_ce, cells_clusters_k_ce]

    def build_cluster_agglomerative_fastcluster(self, d, j):
        """
        This method builds clusters using agglomerative clustering.
        """

        feature_vectors = d.column_features[j]
        clusters_k_c_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        cells_clusters_k_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        try:

            data = feature_vectors
            if(self.FEATUREREDUCTION):
                reduced_feature_vectors = pd.DataFrame(feature_vectors)
                #weighs = reduced_feature_vectors.value_counts(sort=False, normalize=True).to_list()
                reduced_feature_vectors.drop_duplicates(inplace=True)
                data = reduced_feature_vectors.to_numpy()

                if self.VERBOSE:
                    print("Column {}: reduced data from {} to {}".format(j, len(feature_vectors), len(data)))
            
            clustering_model = fastcluster.linkage_vector(data, method="single", metric="hamming")

            for k in clusters_k_c_ce:
                model_labels = [l - 1 for l in
                                scipy.cluster.hierarchy.fcluster(clustering_model, k, criterion="maxclust")]

                if(self.FEATUREREDUCTION):
                    #model_labels = clustering_model.predict(feature_vectors)
                    label_dict = {}
                    for index,row in enumerate(data):
                        label_dict[tuple(row)] = model_labels[index]

                    model_labels = numpy.zeros(len(feature_vectors), dtype=int)
                    for index,feature_vector in enumerate(feature_vectors):
                        model_labels[index] = label_dict[tuple(feature_vector)]


                for index, c in enumerate(model_labels):
                    if c not in clusters_k_c_ce[k]:
                        clusters_k_c_ce[k][c] = {}
                    cell = (index, j)
                    clusters_k_c_ce[k][c][cell] = 1
                    cells_clusters_k_ce[k][cell] = c
        except Exception as e:
            pass

        if self.VERBOSE:
            print("A hierarchical clustering model with single-linkage is built for column {}.".format(j))
        return [clusters_k_c_ce, cells_clusters_k_ce]


    def build_cluster_hdbscan(self, d, j):
        """
        This method builds clusters using hdbscan.
        """
        feature_vectors = d.column_features[j]
        clusters_k_c_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        cells_clusters_k_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        try:
                
            k = self.LABELING_BUDGET+1

            #adjust min_cluster_size and min_samples for dataset
            clustering_model = hdbscan.HDBSCAN(min_cluster_size=self.HDBSCAN_MIN_CLUSTER_SIZE, min_samples=self.HDBSCAN_MIN_SAMPLES, cluster_selection_epsilon=0, cluster_selection_method='eom', core_dist_n_jobs= mp.cpu_count())
            model_labels = clustering_model.fit(feature_vectors).labels_

            for index, c in enumerate(model_labels):
                if c not in clusters_k_c_ce[k]:
                    clusters_k_c_ce[k][c] = {}
                cell = (index, j)
                clusters_k_c_ce[k][c][cell] = 1
                cells_clusters_k_ce[k][cell] = c
        except Exception as e:
            pass

        if self.VERBOSE:
            print("A HDBScan clustering model is built for column {}.".format(j))
        return [clusters_k_c_ce, cells_clusters_k_ce]

    def get_Centroids(feature_vector, n_clusters, column_strategys,random_state=0):
        """
        This method generates cluster centers for k-means and mini-batch k-means.
        """
        ####Generating synthetic feature-vector####
        frame = pd.DataFrame(feature_vector)
        start = None
        sum_start = len(feature_vector[0]) + 1
        zer = numpy.zeros(len(feature_vector[0]))
    
        
        for index,strategy in enumerate(column_strategys):

            
            if strategy == "PVD" or strategy == "TFIDF":
                count = frame[index].value_counts()
                

                # Median without sorting
                v = []
                for ex in count.items():
                    v.append(ex)

                
                if v[0][1] > v[1][1]:
                    val = v[0][0]
                elif v[0][1] < v[1][1]:
                    val = v[1][0]
                else:
                    val = 0.5


                zer[index] = val
            elif strategy == "RVD":
                zer[index] = 1
        

        ####Searching for cell closest to synthetic feature-vector####
        for features in feature_vector:
            
            feature_sum = scipy.spatial.distance.hamming(features, zer)
            if feature_sum < sum_start:
                start = features
                sum_start = feature_sum

                if sum == 0:
                    break;

        ####Using K-Means++ initialization to get all other centroids####
        centroids = [start]
        for i in range(1,n_clusters):
            distance = 0
            next = None
            
            for features in feature_vector:
                
                feature_distance = scipy.spatial.distance.hamming(centroids[-1],features)
                if (feature_distance > distance):
                    next = features
                    distance = feature_distance
            
            centroids.append(next)
        
        return numpy.array(centroids)

    def build_cluster_kmeans(self, d, j):
        """
        This method builds clusters using k-means.
        """
        feature_vectors = d.column_features[j]
        clusters_k_c_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        cells_clusters_k_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}

        try:
            
            data = feature_vectors
            if len(data[0]) < 1:
                raise Exception()
            weighs = numpy.ones(len(feature_vectors))

            # Deduplication of feature-vectors
            if(self.FEATUREREDUCTION):
                reduced_feature_vectors = pd.DataFrame(feature_vectors)
                
                weighs = reduced_feature_vectors.value_counts(sort=False, normalize=True).to_list()
                reduced_feature_vectors.drop_duplicates(inplace=True)
                
                data = reduced_feature_vectors.to_numpy()

                if self.VERBOSE:
                    print("Column {}: reduced data from {} to {}".format(j, len(feature_vectors), len(data)))

            k = self.LABELING_BUDGET+1
            
            nclus = k

            if len(data) < nclus:
                nclus = len(data)

            data = data.astype(int)
            clustering_model = sl.cluster.KMeans(n_clusters=nclus, init= FasterDetection.get_Centroids(data,nclus,d.NAME_INDEX[j]), algorithm='full', n_init=1, max_iter=10000)
            clustering_model.fit(data, sample_weight=weighs)
            model_labels = clustering_model.labels_

            
            # Reduplication of feature-vector for later usage
            if(self.FEATUREREDUCTION):

                label_dict = {}
                for index,row in enumerate(data):
                    label_dict[tuple(row)] = model_labels[index]

                model_labels = numpy.zeros(len(feature_vectors), dtype=int)
                for index,feature_vector in enumerate(feature_vectors):
                    model_labels[index] = label_dict[tuple(feature_vector)]
            
            for index, c in enumerate(model_labels):
                if c not in clusters_k_c_ce[k]:
                    clusters_k_c_ce[k][c] = {}
                
                cell = (index, j)
                clusters_k_c_ce[k][c][cell] = 1
                cells_clusters_k_ce[k][cell] = c
        except Exception as e:
            pass

        if self.VERBOSE:
            print("A K-Means clustering model is built for column {}.".format(j))
        return [clusters_k_c_ce, cells_clusters_k_ce]

    def build_cluster_mbatch(self, d, j):
        """
        This method builds clusters using mini-batch k-means.
        """
        feature_vectors = d.column_features[j]
        clusters_k_c_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        cells_clusters_k_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        try:
                
            data = feature_vectors
            weighs = numpy.ones(len(feature_vectors))

            # deduplication
            if(self.FEATUREREDUCTION):
                reduced_feature_vectors = pd.DataFrame(feature_vectors)
                weighs = reduced_feature_vectors.value_counts(sort=False, normalize=True).to_list()
                reduced_feature_vectors.drop_duplicates(inplace=True)
                data = reduced_feature_vectors.to_numpy()

                if self.VERBOSE:
                    print("Column {}: reduced data from {} to {}".format(j, len(feature_vectors), len(data)))

            k = self.LABELING_BUDGET+1
            
            nclus = k
            
            
            if len(data) < nclus:
                nclus = len(data)

            # adjust batch_size for faster or accuracy
            clustering_model = sl.cluster.MiniBatchKMeans(n_clusters=nclus, init=FasterDetection.get_Centroids(data,nclus, d.NAME_INDEX[j]), max_iter=10000, batch_size=1024, max_no_improvement=10, init_size=int(len(data)* self.MBATCH_SIZE), n_init = 1)
            model_labels = clustering_model.fit(data, sample_weight=weighs).labels_

            #reduplication
            if(self.FEATUREREDUCTION):
                #model_labels = clustering_model.predict(feature_vectors)
                label_dict = {}
                for index,row in enumerate(data):
                    label_dict[tuple(row)] = model_labels[index]

                model_labels = numpy.zeros(len(feature_vectors), dtype=int)
                for index,feature_vector in enumerate(feature_vectors):
                    model_labels[index] = label_dict[tuple(feature_vector)]
            
            for index, c in enumerate(model_labels):
                if c not in clusters_k_c_ce[k]:
                    clusters_k_c_ce[k][c] = {}
                
                cell = (index, j)
                clusters_k_c_ce[k][c][cell] = 1
                cells_clusters_k_ce[k][cell] = c
        except Exception as e:
            pass

        if self.VERBOSE:
            print("A Mini-Batch-K-Means clustering model is built for column {}.".format(j))
        return [clusters_k_c_ce, cells_clusters_k_ce]

    def build_cluster_kmodes(self, d, j):
        """
        This method builds clusters using k-modes.
        """
        feature_vectors = d.column_features[j]
        clusters_k_c_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        cells_clusters_k_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        try:
            
            data = feature_vectors

            k = self.LABELING_BUDGET+1
            nclus = k
            
            
            clustering_model = km.KModes(n_clusters=nclus, init="Huang", n_init=1, n_jobs=1, verbose=0)
            model_labels = clustering_model.fit_predict(data)

            
            for index, c in enumerate(model_labels):
                if c not in clusters_k_c_ce[k]:
                    clusters_k_c_ce[k][c] = {}
                
                cell = (index, j)
                clusters_k_c_ce[k][c][cell] = 1
                cells_clusters_k_ce[k][cell] = c
        except Exception as e:
            pass

        if self.VERBOSE:
            print("A K-Modes clustering model is built for column {}.".format(j))
        return [clusters_k_c_ce, cells_clusters_k_ce]
    
    def build_cluster_parc(self, d, j):
        """
        This method builds clusters using parc.
        """
        feature_vectors = d.column_features[j]
        clusters_k_c_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        cells_clusters_k_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        try:
                
            k = self.LABELING_BUDGET+1
            
            # adjust dist_std_local and jac_std_global for dataset
            clustering_model = parc.PARC(feature_vectors, dist_std_local=self.PARC_DIST_STD_LOCAL, jac_std_global=self.PARC_JAC_STD_GLOBAL, jac_weighted_edges=True, resolution_parameter=1) 
            clustering_model.run_PARC()
            model_labels = clustering_model.labels
            
            for index, c in enumerate(model_labels):
                if c not in clusters_k_c_ce[k]:
                    clusters_k_c_ce[k][c] = {}
                
                cell = (index, j)
                clusters_k_c_ce[k][c][cell] = 1
                cells_clusters_k_ce[k][cell] = c
        except Exception as e:
            pass

        if self.VERBOSE:
            print("A PARC clustering model is built for column {}.".format(j))
        return [clusters_k_c_ce, cells_clusters_k_ce]

    def build_cluster_birch(self, d, j):
        """
        This method builds clusters using birch and kmeans.
        """
        feature_vectors = d.column_features[j]
        clusters_k_c_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        cells_clusters_k_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
        try:
            data = feature_vectors
            
            #calculate threshold for birch
            if self.BIRCH_THRESH == -1:
                reduced_feature_vectors = pd.DataFrame(feature_vectors)
                reduced_feature_vectors = reduced_feature_vectors.drop_duplicates(inplace=False)
                thresh = len(reduced_feature_vectors) / len(feature_vectors)
            else:
                tresh = self.BIRCH_THRESH

            k = self.LABELING_BUDGET+1
            nclus = k
            
            
            if len(data) < nclus:
                nclus = len(data)
            
            # if needed threshhold can be adjusted
            clustering_model = sl.cluster.Birch(n_clusters=None, branching_factor=50, threshold=thresh)
            model_labels = clustering_model.fit_predict(data)
            subcluster_center = clustering_model.subcluster_centers_

            if len(subcluster_center) < nclus:
                nclus = len(subcluster_center)

            global_model = sl.cluster.KMeans(n_clusters=nclus, init=FasterDetection.get_Centroids(data,nclus, d.NAME_INDEX[j]), algorithm='full', n_init=1, max_iter=10000)
            
            subcluster_labels = global_model.fit_predict(subcluster_center)

            for index,cluster in enumerate(model_labels):
                model_labels[index] = subcluster_labels[cluster]
            
            for index, c in enumerate(model_labels):
                if c not in clusters_k_c_ce[k]:
                    clusters_k_c_ce[k][c] = {}
                
                cell = (index, j)
                clusters_k_c_ce[k][c][cell] = 1
                cells_clusters_k_ce[k][cell] = c
        except Exception as e:
            pass

        if self.VERBOSE:
            print("A Birch clustering model is built for column {}.".format(j))
        return [clusters_k_c_ce, cells_clusters_k_ce]

    def build_clusters_in_parallel(self, d):
        """
        This method parallelized the clustering of the dataset columns.
        """
        algorithm = self.CLUSTER_ALGORITHM
        if(algorithm == self.algorithms[0]):

            function = self.build_cluster_kmeans
        elif (algorithm == self.algorithms[1]):

            function = self.build_cluster_mbatch
        elif (algorithm == self.algorithms[2]):
                
            function = self.build_cluster_hdbscan
        elif (algorithm == self.algorithms[3]):

            function = self.build_cluster_agglomerative
        elif (algorithm == self.algorithms[4]):

            function = self.build_cluster_kmodes
        elif (algorithm == self.algorithms[5]):

            function = self.build_cluster_parc
        elif (algorithm == self.algorithms[6]):

            function = self.build_cluster_birch
        elif (algorithm == self.algorithms[7]):

            function = self.build_cluster_agglomerative_fastcluster

        executor = Parallel(n_jobs=self.N_JOBS ,prefer="processes")
        tasks = (delayed(function)(d, j) for j in range(d.dataframe.shape[1]))
        clustering_results = executor(tasks)
        
        d.clusters_k_j_c_ce = {k: {j: clustering_results[j][0][k] for j in range(d.dataframe.shape[1])} for k in
                               range(2, self.LABELING_BUDGET + 2)}
        d.cells_clusters_k_j_ce = {k: {j: clustering_results[j][1][k] for j in range(d.dataframe.shape[1])} for k in
                                   range(2, self.LABELING_BUDGET + 2)}

    #modified sample_tuple
    def alternative_sample_tuple(self, d):
        """
        This method samples a tuple for cluster algorithms that do not build clusters iterativly
        """
        # --------------------Calculating Number of Labels per Clusters--------------------

        ##### 
        k = self.LABELING_BUDGET + 1
        #####

        for j in range(d.dataframe.shape[1]):
            for c in d.clusters_k_j_c_ce[k][j]:
                d.labels_per_cluster[(j, c)] = {cell: d.labeled_cells[cell][0] for cell in d.clusters_k_j_c_ce[k][j][c] if
                                                cell[0] in d.labeled_tuples}
        # --------------------Sampling a Tuple--------------------
        if self.CLUSTERING_BASED_SAMPLING:
            tuple_score = numpy.zeros(d.dataframe.shape[0])
            cluster_vector = {}
            for i in range(d.dataframe.shape[0]):
                cluster_vector[i] = []
                if i not in d.labeled_tuples:
                    score = 0.0
                    for j in range(d.dataframe.shape[1]):
                        if d.clusters_k_j_c_ce[k][j]:
                            cell = (i, j)
                            c = d.cells_clusters_k_j_ce[k][j][cell]
                            cluster_vector[i].append(c)
                            score += math.exp(-len(d.labels_per_cluster[(j, c)]))  
                    tuple_score[i] = math.exp(score)
            
        else:
            tuple_score = numpy.ones(d.dataframe.shape[0])
        sum_tuple_score = sum(tuple_score)
        p_tuple_score = tuple_score / sum_tuple_score
        
        d.sampled_tuple = numpy.random.choice(numpy.arange(d.dataframe.shape[0]), 1, p=p_tuple_score)[0]
        if self.VERBOSE:
            print("Tuple {} is sampled.".format(d.sampled_tuple))

    # new run method
    
    def run(self, dd):
        """
        This method runs Raha on an input dataset to detection data errors.
        """
        #algorithms = ['kmeans', 'mbatch', 'hdbscan', 'agglomerative', 'kmodes', 'parc', 'birch']
        algorithm = self.CLUSTER_ALGORITHM
        if algorithm not in self.algorithms:
            raise ValueError("Your choosen algorithm does not exist or is not implemented, please choose from {}".format(self.algorithms))


        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "---------------------Initializing the Dataset Object--------------------\n"
                  "------------------------------------------------------------------------")
        d = self.initialize_dataset(dd)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-------------------Running Error Detection Strategies-------------------\n"
                  "------------------------------------------------------------------------")
        self.run_strategies(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-----------------------Generating Feature Vectors-----------------------\n"
                  "------------------------------------------------------------------------")
        self.generate_features(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------------Building the Clustering Model-----------------------\n"
                  "------------------------------------------------------------------------")
        self.build_clusters_in_parallel(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-------------Iterative Clustering-Based Sampling and Labeling-----------\n"
                  "------------------------------------------------------------------------")

        sample_tuple = self.alternative_sample_tuple
        if algorithm in ['agglomerative_average', 'agglomerative_single']:
            if self.VERBOSE:
                print("using old sample_tupel")
            sample_tuple = self.sample_tuple
        
        while len(d.labeled_tuples) < self.LABELING_BUDGET:

            sample_tuple(d)
            if d.has_ground_truth:
                self.label_with_ground_truth(d)
            
            # else:
            #   In this case, user should label the tuple interactively as shown in the Jupyter notebook.
            if self.VERBOSE:
                print("------------------------------------------------------------------------")

        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------Propagating User Labels Through the Clusters--------------\n"
                  "------------------------------------------------------------------------")
        self.propagate_labels(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "---------------Training and Testing Classification Models---------------\n"
                  "------------------------------------------------------------------------")
        self.predict_labels(d)
        if self.SAVE_RESULTS:
            if self.VERBOSE:
                print("------------------------------------------------------------------------\n"
                      "---------------------------Storing the Results--------------------------\n"
                      "------------------------------------------------------------------------")
            self.store_results(d)
        return d.detected_cells
########################################


########################################
if __name__ == "__main__":

    dataset_name = "hospital"
    dataset_dictionary = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
        "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
    }
    app = FasterDetection()
    app.VERBOSE = True
    app.LABELING_BUDGET = 20
    app.FEATUREREDUCTION = False
    app.N_JOBS = 1
    app.CLUSTER_ALGORITHM = 'kmeans'
    
    detection_dictionary = app.run(dataset_dictionary)
    
    data = rh.dataset.Dataset(dataset_dictionary)
    p, r, f = data.get_data_cleaning_evaluation(detection_dictionary)[:3]
    print("Raha's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(data.name, p, r, f))
