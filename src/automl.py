from clustering import Clustering

def autoML(clust, methods):
    for method in methods:
        clust.load_method(method)
        print(method)
        clust.clusterize(method, t=300)
    results = clust.get_results()
    print(results)
        