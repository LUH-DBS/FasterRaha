
import os

import matplotlib.pyplot as plt
import numpy
from sklearn.manifold import TSNE

import fasterdetection as craha


### change here to generate visualization for specific algortihm ###
algorithm = 'kmeans'
algorithms = ['kmeans', 'mbatch', 'hdbscan', 'agglomerative', 'kmodes', 'parc', 'birch']

dataset = "hospital"
dataset_dictionary = {
    "name": dataset,
    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset, "dirty.csv")),
    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset, "clean.csv"))
}

pa = os.path.join(os.getcwd(), "Visualization")
if not os.path.isdir(pa):
    os.mkdir(pa)

pa_algo = os.path.join(pa, algorithm)
if not os.path.isdir(pa_algo):
    os.mkdir(pa_algo)

pa_data = os.path.join(pa_algo, dataset)
if not os.path.isdir(pa_data):
    os.mkdir(pa_data)

app = craha.FasterDetection()
app.CLUSTER_ALGORITHM = algorithm

d = app.initialize_dataset(dataset_dictionary)
app.run_strategies(d)
app.generate_features(d)

n_jobs = 1
app.build_clusters_in_parallel(d)

k = app.LABELING_BUDGET + 1

for j in range(d.dataframe.shape[1]):
    features = d.column_features[j]

    if(len(features[0]) == 0):
        continue

    
    dirty = d.dataframe.items()
    clean = d.clean_dataframe.items()


    ### creating cluster from ground truth ###
    truth = []
    for index,(dirty_column, clean_column) in enumerate(zip(dirty, clean)):
        if index == j:
            truth = numpy.where((dirty_column[1] == clean_column[1]), "#0000FF", "#FF0000")


    ### reduce data dimensionality ###
    tsne = TSNE(n_jobs=-1, perplexity=30, random_state=42)
    tsnefeatures = tsne.fit_transform(features)

    x = [point[0] for point in tsnefeatures]
    y = [point[1] for point in tsnefeatures]


    ### create ground truth plot ###
    plt.scatter(x,y, c=truth)
    plt.savefig('{}/truth{}.eps'.format(pa_data,j),format='eps',bbox_inches = "tight")
    plt.close()

    print("Created truth visualization of Column {}".format(j))


    ### create visualization of cluster algorithm ###
    feat_copy = []
    for index,columns in enumerate(features):
        cell = (index, j)
        feat_copy.append(d.cells_clusters_k_j_ce[k][j][cell])

    plt.scatter(x,y, c=feat_copy)
    plt.savefig('{}/column{}.eps'.format(pa_data, j),format='eps',bbox_inches = "tight")
    plt.close()

    print("Created cluster visualization of Column {}".format(j))
