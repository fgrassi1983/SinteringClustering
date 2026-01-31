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

    def _fast_min_dist(self, points_a, points_b):
        """Efficiently compute minimum distance between two point sets."""
        n_a, n_b = len(points_a), len(points_b)
        if n_a * n_b <= 2000:
            return cdist(points_a, points_b).min()
        tree_b = BallTree(points_b, metric=self.metric)
        distances, _ = tree_b.query(points_a, k=1)
        return distances.min()

    def _fast_median(self, arr):
        """Fast median using partition instead of full sort."""
        n = len(arr)
        mid = n // 2
        if n % 2 == 0:
            partitioned = np.partition(arr, [mid - 1, mid])
            return (partitioned[mid - 1] + partitioned[mid]) / 2.0
        return np.partition(arr, mid)[mid]

    def clusters_can_be_merged(self, points_a, points_b, intra_distances_a, intra_distances_b):
        """Optimized merge check - same logic as original."""
        shortest_dist = self._fast_min_dist(points_a, points_b)
        combined = np.concatenate((intra_distances_a, intra_distances_b))
        # Handle empty combined array (matches original NaN behavior -> returns False)
        if len(combined) == 0:
            return False
        threshold = self._fast_median(combined) * self.coeff
        return shortest_dist <= threshold

    def merge_clusters(self):
        """Optimized cluster merging - matches original logic, avoids dict reconstruction."""
        # Convert to lists to avoid per-merge dict reconstruction
        cluster_keys = list(self.clusters.keys())
        cluster_vals = [self.clusters[k] for k in cluster_keys]
        distance_vals = [self.intra_cluster_distances[k] for k in cluster_keys]

        # Pre-compute point arrays (will update on merge)
        point_arrays = [self.points.take(list(c), axis=0) for c in cluster_vals]

        current_cluster = 0
        run = True

        while run:
            n_clusters_before = len(cluster_vals)

            while current_cluster < len(cluster_vals):
                # Check all other clusters (same order as original)
                merged = False
                for other_cluster in range(len(cluster_vals)):
                    if other_cluster == current_cluster:
                        continue

                    c0 = point_arrays[current_cluster]
                    c1 = point_arrays[other_cluster]
                    d0 = distance_vals[current_cluster]
                    d1 = distance_vals[other_cluster]

                    if self.clusters_can_be_merged(c0, c1, d0, d1):
                        # Merge into current_cluster
                        cluster_vals[current_cluster] = cluster_vals[current_cluster].union(
                            cluster_vals[other_cluster]
                        )
                        distance_vals[current_cluster] = distance_vals[current_cluster] + distance_vals[other_cluster]

                        # Update point array cache
                        point_arrays[current_cluster] = self.points.take(
                            list(cluster_vals[current_cluster]), axis=0
                        )

                        # Remove other_cluster
                        del cluster_vals[other_cluster]
                        del distance_vals[other_cluster]
                        del point_arrays[other_cluster]

                        merged = True
                        break

                current_cluster += 1

            run = n_clusters_before > len(cluster_vals)

        # Rebuild dicts once at the end
        self.clusters = {i: cluster for i, cluster in enumerate(cluster_vals)}
        self.intra_cluster_distances = {i: dist for i, dist in enumerate(distance_vals)}

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
