from clustering import Clustering
from sklearn.metrics import confusion_matrix, calinski_harabasz_score, silhouette_score, davies_bouldin_score

def autoML(clust, methods, checkpoint):
    for method in methods:
        clust.load_method(method)
        #print(method)
        clust.clusterize(method, t=checkpoint)
    results = clust.get_results()
    print("\n\n======", "results", "======\n\n")

    db_results_sorted_by_best = {k: v for k, v in sorted(results["DB"].items(), key=lambda item:item[1], reverse=False)}
    ch_results_sorted_by_best = {k: v for k, v in sorted(results["CH"].items(), key=lambda item:item[1], reverse=True)}
    ac = {k: v for k, v in sorted(results["AC"].items(), key=lambda item:item[1], reverse=True)}
    print("Davies Bouldin: ", db_results_sorted_by_best)
    print("Calinski Harabasz: ", ch_results_sorted_by_best)
    print("Accuracy: ", ac)