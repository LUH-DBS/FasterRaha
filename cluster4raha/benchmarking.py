import argparse as parse
import os
import sys
import time

import memory_profiler
import numpy as np
import psutil
import raha as rh

import fasterdetection as craha

parser = parse.ArgumentParser(description='Benchmarks different algorithms implemented in Raha')
parser.add_argument('algorithms', metavar='algorithm', type=str, nargs='+', help='the algorithms to be benchmarked')
parser.add_argument('--big', action='store_true', help='If given, then the tax dataset is also computed')
parser.add_argument('--iter', default=10 ,metavar='iterations', type=int, help='iterations that will be evaluated')
parser.add_argument('--verbose', action='store_true', help='If given, more progress indicators will be printed')
parser.add_argument('--n_jobs', default=1, metavar='Jobs', type=int, help='Workers for running cluster calculations of columns in parallel')
parser.add_argument('--datasets', default=["rayyan", "hospital", "beers", "movies_1","flights"], metavar='dataset', type=str, nargs='+', help='all datasets that should be benchmarked')
parser.add_argument('--reduction', action='store_true', help='If given, feature reduction will be used')


args = parser.parse_args()

print(psutil.cpu_count(), psutil.virtual_memory())

datasets = args.datasets
#["rayyan", "hospital", "beers", "movies_1","flights"]
if args.big:
    datasets.insert(0, "tax")


pa = os.path.join(os.getcwd(), "Benchmarks")
if not os.path.isdir(pa):
    os.mkdir(pa)


runs = len(args.algorithms) * len(datasets) * args.iter
act_run = 1
for algorithm in args.algorithms:

    pa_algo = os.path.join(pa, algorithm)
    if not os.path.isdir(pa_algo):
        os.mkdir(pa_algo)
    
    performance = []

    for index,dataset in enumerate(datasets):

        performance.append([])

        for i in range(args.iter):
            dd = {
            "name": dataset,
            "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset, "dirty.csv")),
            "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset, "clean.csv"))
            }

            app = craha.FasterDetection()
        
            ### adjust settings here ###
            app.FEATUREREDUCTION = args.reduction
            app.LABELING_BUDGET = 20
            app.N_JOBS = args.n_jobs
            app.CLUSTER_ALGORITHM = algorithm
            # app.MBATCH_SIZE = 0.6
            # app.BIRCH_THRESH = -1
            # app.HDBSCAN_MIN_CLUSTER_SIZE = 2
            # app.HDBSCAN_MIN_SAMPLES = 100
            # app.PARC_DIST_STD_LOCAL = 4
            # app.PARC_JAC_STD_GLOBAL = 0.15
            ###########################

            

            if algorithm not in app.algorithms:
                sys.exit("Your choosen algorithm does not exist or is not implemented, please choose from {}".format(app.algorithms))


            
            d = app.initialize_dataset(dd)
            app.run_strategies(d)
            
            app.generate_features(d)
            

            psutil.cpu_percent(interval=None)
            begin = time.time()
            
            mem = memory_profiler.memory_usage((app.build_clusters_in_parallel, (d, ), ))

            end = time.time()
            cpu_usage_after = psutil.cpu_percent(interval=None)

            sample_tuple = app.alternative_sample_tuple
            if algorithm in ['agglomerative', 'fastcl']:
                if app.VERBOSE:
                    print("using old sample_tupel")
                sample_tuple = app.sample_tuple
            
            while len(d.labeled_tuples) < app.LABELING_BUDGET:

                sample_tuple(d)
                #self.sample_tuple(d)
                if d.has_ground_truth:
                    app.label_with_ground_truth(d)
                # else:
                #   In this case, user should label the tuple interactively as shown in the Jupyter notebook
            
            app.propagate_labels(d)

            app.predict_labels(d)
            
            app.store_results(d)

            detection_dictionary = d.detected_cells
            
            
            data = rh.dataset.Dataset(dd)
            p, r, f = data.get_data_cleaning_evaluation(detection_dictionary)[:3]

            performance[index].append([p, r, f, end-begin, cpu_usage_after, max(mem)])
            #print("Raha's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(data.name, p, r, f))

            if args.verbose:
                print("\nRun {} / {} : {} with {}\n".format(act_run, runs, dataset, algorithm))
            act_run = act_run + 1

    avg_per_str_out = ""    
    for per, dataset in zip(performance,datasets):
        file = open(os.path.join(pa_algo, "{}.bench".format(dataset)),  "w")
        file.write(str(per))
        file.close()
        
        performance_np = np.array(per)
        avg_array = np.mean(performance_np, axis=0)

        avg_per_str = "Rahas's performance in average on {} with {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}\nTime = {:.5f} s\nCPU Usage: {:.2f}%\nRAM Usage = {:.4f} MB\n".format(dataset, algorithm, avg_array[0], avg_array[1], avg_array[2], avg_array[3], avg_array[4], avg_array[5])
        print(avg_per_str)

        avg_per_str_out += "\n" +avg_per_str
        
    file = open(os.path.join(pa_algo, "average.bench"),  "w")
    file.write(avg_per_str_out)
    file.close()
