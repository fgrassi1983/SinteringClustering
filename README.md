# SinteringClustering
Custom made distance-based clustering algorithm

Developed by: Fabio Grassi, PhD
Debugged by: Paolo Sofia

## How it works

The Sintering algorithm works by first grouping neighbouring points into clusters. Each cluster K0 starts by taking one point P0 and finding its closest neighbour P1. After that, all points within radius r = distance(P0,P1) * coeff are added to K0. The process is repeated for all newly added points until no new neighbours are found, then a new cluster K1 is created.

After this initial step, the algorithm proceeds to merge clusters together, based on the following criterion: given two clusters Km and Kn, the distance D between the clusters is defined as the minimum distance between each point in Pm and each point in Pn. If this distance is less than or equal to the median of the distance between each neighbour inside the two clusters, multiplied by coeff, then the clusters are merged.
