# FasterRaha
Collection of different cluster algorithms integrated in Raha to improve scalability.

To improve the scalability of Raha mulitple cluster algorithms have been implemented. The usage of the cluster algorithms was parallelized. On top, it is now possible to eliminate duplicates to decrease runtime. 

# Usage

It is recommended to use k-means or hierarchical agglomerative clustering with single-linkage with feature-reduction. If you have more knowlege about your dataset, you are open to use the other implemented clustering algorithms.


* <code>cluster4raha/fasterdetection.py</code>
  * the class "FasterDetection" is designed to replace the class "Detection" from Raha.
  * <code>FasterDetection.CLUSTER_ALGORITHM </code> defines the clustering algorithm that is used.
  * <code>FasterDetection.FEATUREREDUCTION</code> determines if the de- and reduplication is activated.
  * <code>FasterDetection.N_JOBS</code> determines how many cores are used for high-level parallelisation.
        
  * all cluster algorithms can be configured with: <br>
    <pre><code>FasterDetection.MBATCH_SIZE
    FasterDetection.BIRCH_THRESH
    FasterDetection.HDBSCAN_MIN_CLUSTER_SIZE
    FasterDetection.HDBSCAN_MIN_SAMPLES
    FasterDetection.PARC_DIST_STD_LOCAL
    FasterDetection.PARC_JAC_STD_GLOBAL</code></pre>
  
* <code>cluster4raha/benchmarks.py</code>
  * this file contains a script which can be used benchmark FasterDetection. It will generate an output directory which contains output files. output files that are called <code>average.bench</code> contain the averaged results of the benchmarks. All other files store the measured values for each iteration on one dataset. 
  * there are several arguments that can be given
    * <code>algorithms </code> : determines which algorithm should be benchmarked. More than one can be given. ['kmeans', 'mbatch', 'hdbscan', 'agglomerative', 'kmodes', 'parc', 'birch', 'agglomerative_single']
    * <code>--big</code> (optional) (default=false) : adds the tax dataset to the datasets that are benchmarked
    * <code>--iter</code> (optional) (default = 10) : sets how many iterations are run for the benchmark
    * <code>--verbose</code> (optional) (default = false) : adds a counter for the iterations
    * <code>--n_jobs</code> (optional) (default = 1) : sets how many cores are used for high-level parallelisation
    * <code>--datasets</code> (optional) (default = all) : sets which datasets the benchmark should be run on. More than one can be given. ["rayyan", "hospital", "beers", "movies_1","flights"]
    * <code>--reduction</code> (optional) (default = false) : activates the deduplication of the dataset. Is only useful for K-Means and hierarchical agglomerative clustering with single-linkage

* <code>cluster4raha/visualization.py</code>
  * generates a visualisation of the generated cluster on a dataset and a visualisation for the ground truth values by using t-SNE.
    * which algorithm and dataset needs to be set inside the code

# References

FasterRaha is based upon and uses code from [Raha](https://github.com/BigDaMa/raha).

