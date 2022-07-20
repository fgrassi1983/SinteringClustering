import numpy as np
from scipy.spatial.distance import cdist, pdist
from itertools import product

class Sintering:
    def __init__(self, coeff=3, metric="euclidean"):
        self.metric = metric # Expected values: any metric accepted by scipy.pdist, scipy.cdist
        self.coeff = coeff
    
    def get_one_distance_n(self,n0,n1,return_n0=False):
        ndx = sum([self.N-i for i in range(1,n0+1)]) + n1 - n0 - 1
        return n0 if return_n0 else n1, self.distances[ndx]

    def get_all_distances_n(self,n):
        return np.array(
            [
                self.get_one_distance_n(i,n,return_n0=True)
                for i in range(n)
            ] + [
                self.get_one_distance_n(n,i)
                for i in range(n+1,self.N)
            ]
        )

    def get_neighbours(self,n,coeff):
        point_distances = self.get_all_distances_n(n)
        argmin = np.argmin(point_distances[:,1])
        radius = coeff * point_distances[argmin,1]

        return np.array(
            [
                i
                for i in self.get_all_distances_n(n)
                if i[1] <= radius
            ]
        )
    
    def clusters_can_be_merged(self,points_a,points_b,intra_distances_a,intra_distances_b):
        shortest_dist = cdist(points_a, points_b).min()
        threshold = np.min(
            np.median(
                np.concatenate(
                    (
                        intra_distances_a
                        ,intra_distances_b
                    )
                )
            )
        ) * self.coeff
        return shortest_dist <= threshold
    
    def merge_clusters(self):
        run = 1
        
        new_clusters = self.clusters.copy()
        new_distances = self.intra_cluster_distances.copy()
        
        current_cluster = 0
        while run:
            n_clusters_before = len(new_clusters)
            while current_cluster <= len(new_clusters)-1:
                new_clusters = {i:j for i,j in enumerate(new_clusters.values())}
                new_distances = {i:j for i,j in enumerate(new_distances.values())}
                other_clusters = list(filter(lambda x: x != current_cluster, new_clusters.keys()))
                other_cluster_ndx = 0
                while other_cluster_ndx < len(other_clusters)-1:
                    other_cluster = other_clusters[other_cluster_ndx]
                    c0 = self.points.take(list(new_clusters[current_cluster]),axis=0)
                    c1 = self.points.take(list(new_clusters[other_cluster]),axis=0)
                    d0 = new_distances[current_cluster]
                    d1 = new_distances[other_cluster]
                    if self.clusters_can_be_merged(c0,c1,d0,d1):
                        new_clusters[current_cluster] = new_clusters[current_cluster].union(new_clusters[other_cluster])
                        new_distances[current_cluster].extend(new_distances[other_cluster])
                        del new_clusters[other_cluster]
                        del new_distances[other_cluster]
                        other_clusters = list(filter(lambda x: x != current_cluster, new_clusters.keys()))
                        break
                    other_cluster_ndx += 1
                current_cluster += 1
            self.clusters = new_clusters
            self.intra_cluster_distances = new_distances
            run = n_clusters_before - len(new_clusters)
    
    def sinter(self):
        while True:
            nclusters = len(self.clusters)
            self.merge_clusters()
            if len(self.clusters) == nclusters:
                break
        
        return np.array(
        sorted(
            [
                i
                for j in
                [
                    list(
                        product(
                        [k],self.clusters[k]
                        )
                    )
                    for k in self.clusters
                ]
                for i in j
            ]
            ,key=lambda x: x[1]
        )
    )[:,0]
    
    def fit_predict(self, points):
        self.N = len(points)
        self.points = np.array(points)
        self.distances = pdist(self.points,metric=self.metric)
        self.clusters = {0:{0}}
        self.todo = {i for i in range(self.N)}
        self.intra_cluster_distances = {0:[]}
        current_cluster = -1
        new_cluster = True
        
        points_to_add = set()
        
        while len(self.todo):
            new_points = set()
            
            if len(points_to_add):
                for point in points_to_add:
                    self.clusters[current_cluster].add(point)
                    coeff = 1 if len(self.clusters[current_cluster]) == 1 else self.coeff
                    neighbours = self.get_neighbours(point,coeff)
                    for n, dist in neighbours:
                        new_point = int(n)
                        if new_point in self.todo and new_point not in points_to_add:
                            new_points.add(int(n))
                            self.intra_cluster_distances[current_cluster].append(dist)
                    self.todo.remove(point)
                points_to_add = new_points
                
            else:
                current_cluster +=1
                point = self.todo.pop()
                self.todo.add(point)
                self.clusters[current_cluster] = {point}
                self.intra_cluster_distances[current_cluster] = []
                points_to_add = {point}
        
        return self.sinter()
