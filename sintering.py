import numpy as np
from sklearn.neighbors import BallTree
from scipy.spatial.distance import cdist
from itertools import product

class SinteringOptimized:
    def __init__(self, coeff=3, metric="euclidean"):
        self.coeff = coeff
        self.metric = metric
    
    def get_neighbours(self, point_index, tree, radius):
        ind, dist = tree.query_radius([self.points[point_index]], r=radius, return_distance=True)
        return ind[0], dist[0]
    
    def clusters_can_be_merged(self, points_a, points_b, intra_distances_a, intra_distances_b):
        shortest_dist = cdist(points_a, points_b).min()
        threshold = np.min(
            np.median(
                np.concatenate((intra_distances_a, intra_distances_b))
            )
        ) * self.coeff
        return shortest_dist <= threshold
    
    def merge_clusters(self):
        new_clusters = self.clusters.copy()
        new_distances = self.intra_cluster_distances.copy()
        current_cluster = 0
        run = True
        
        while run:
            n_clusters_before = len(new_clusters)
            while current_cluster < len(new_clusters):
                other_clusters = list(filter(lambda x: x != current_cluster, new_clusters.keys()))
                for other_cluster in other_clusters:
                    c0 = self.points.take(list(new_clusters[current_cluster]), axis=0)
                    c1 = self.points.take(list(new_clusters[other_cluster]), axis=0)
                    d0 = new_distances[current_cluster]
                    d1 = new_distances[other_cluster]
                    
                    if self.clusters_can_be_merged(c0, c1, d0, d1):
                        new_clusters[current_cluster] = new_clusters[current_cluster].union(new_clusters[other_cluster])
                        new_distances[current_cluster].extend(new_distances[other_cluster])
                        del new_clusters[other_cluster]
                        del new_distances[other_cluster]
                        new_clusters = {i: cluster for i, cluster in enumerate(new_clusters.values())}
                        new_distances = {i: dist for i, dist in enumerate(new_distances.values())}
                        break
                current_cluster += 1
            
            run = n_clusters_before > len(new_clusters)
        
        self.clusters = new_clusters
        self.intra_cluster_distances = new_distances
    
    def sinter(self):
        while True:
            nclusters = len(self.clusters)
            self.merge_clusters()
            if len(self.clusters) == nclusters:
                break
        
        return np.array(
            sorted(
                [i for j in [
                    list(product([k], self.clusters[k])) for k in self.clusters
                ] for i in j], key=lambda x: x[1]
            )
        )[:, 0]
    
    def fit_predict(self, points):
        self.points = np.array(points)
        self.N = len(self.points)
        self.tree = BallTree(self.points, metric=self.metric)
        self.clusters = {0: {0}}
        self.todo = set(range(self.N))
        self.intra_cluster_distances = {0: []}
        current_cluster = -1
        points_to_add = set()

        while len(self.todo):
            new_points = set()
            
            if points_to_add:
                for point in points_to_add:
                    self.clusters[current_cluster].add(point)
                    coeff = 1 if len(self.clusters[current_cluster]) == 1 else self.coeff
                    neighbours, dists = self.get_neighbours(point, self.tree, coeff * np.min(self.intra_cluster_distances[current_cluster]) if self.intra_cluster_distances[current_cluster] else 1.0)
                    for new_point, dist in zip(neighbours, dists):
                        if new_point in self.todo and new_point not in points_to_add:
                            new_points.add(new_point)
                            self.intra_cluster_distances[current_cluster].append(dist)
                    self.todo.remove(point)
                points_to_add = new_points
            
            else:
                current_cluster += 1
                point = self.todo.pop()
                self.todo.add(point)
                self.clusters[current_cluster] = {point}
                self.intra_cluster_distances[current_cluster] = []
                points_to_add = {point}
        
        return self.sinter()
