from clustering import Clustering
from automl import autoML
import pandas as pd
import time
from dataloader import load_c51
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from sklearn.metrics import confusion_matrix, calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_som.som import SOM
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
c51path = "/home/luis/Documents/git_rep/DBSCAN_Experimentation/data/capture51/csvs/capture_51_csv_parts/"
#data = pd.read_csv("c52_packets_csv.csv")
#data = pd.read_csv("CIC_packets_csv.csv")
early_warning_url = "../../wgrs-data-2023/data/exp_1.csv"
early_warning = pd.read_csv(early_warning_url, sep=";")


###data , infected = load_c51(c51path)
#data , infected = load_c51("../../DBSCAN_Experimentation/data/CIC/csvs/")

colunas = ['5_kurt_total_ips_origem', '5_skw_total_ips_destino', '5_coefficient_variation_total_pacotes' ]
limit = 542
init = 0
x_train = early_warning[colunas].copy()[init:limit]
y_real = early_warning['has_bot'][init:limit]
x_train = x_train.fillna(0)
clustering = KMeans(n_clusters=2, random_state=0).fit(x_train)    
y_test = clustering.labels_
cb = calinski_harabasz_score(x_train, y_test)
db = davies_bouldin_score(x_train, y_test)
print(cb)
print(db)
print(accuracy_score(y_real, y_test))
#print(confusion_matrix(y_real, y_test))
print(classification_report(y_real, y_test, digits=4))
ConfusionMatrixDisplay.from_predictions(y_real, y_test)

#clust = Clustering(data, dataset="capture51", infected=infected)
#clust = Clustering(data, dataset="cic", infected=infected)
#clust = Clustering(data, dataset="capture52", infected=[])

clust = Clustering(x_train, dataset="capture51", infected=y_real, preprocess=0)

methods = ["KMeans", "DBSCAN", "SOM", "Birch", "Ward", "Spectral"]

c51_checkpoints={
    "udp_1": 5632,
    "udp_end_1": 6508,
    "udp_2": 6581,
    "udp_end_2": 6772,
    "udp_3": 6798,
    "udp_end_3": 6951,
    "icmp_1": 7153,
    "icmp_end_1": 7949,
}

#clust.ground_truth("c51", t=7949)
autoML(clust, methods, 542)
'''
    for chkpnt, time_stamp in c51_checkpoints.items():
        t = time.time()
        clust.load_method(method)
        clust.clusterize(chkpnt, t=time_stamp)
        clust.ground_truth(chkpnt, t=time_stamp)
        t_end = time.time()
        print("Elapsed {} in {}: {}".format(chkpnt, method, t_end-t))
    #clust.clusterize("Breakpoint", t=778)
    #clust.ground_truth("Breakpoint", t=778)
    #clust.clusterize("Attack2", t=778+29)
    #clust.ground_truth("Attack2", t=778+29)
    #clust.clusterize("End", t=903)
    #clust.ground_truth("End", t=903)
'''