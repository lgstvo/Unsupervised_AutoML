from clustering import Clustering

def autoML(clust, methods, checkpoint):
    for method in methods:
        clust.load_method(method)
        #print(method)
        clust.clusterize(method, t=checkpoint)
    results = clust.get_results()
    print("\n\n\n\n======", results, "======\n\n\n\n")

    db_results_sorted_by_best = {k: v for k, v in sorted(results["DB"].items(), key=lambda item:item[1], reverse=False)}
    ch_results_sorted_by_best = {k: v for k, v in sorted(results["CH"].items(), key=lambda item:item[1], reverse=True)}
    print(db_results_sorted_by_best)
    print(ch_results_sorted_by_best)
    