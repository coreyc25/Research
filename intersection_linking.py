"""
4) Intersection Linking: This approach focuses on detecting intersections
 first and then linking them together using trajectory information[7].
Karagiorgou [11] used changes in the heading to
identify their intersection points.
The outline of this process is as follows: 
1. Extract points from input trajectories that satisfy specific heading
(turning) or speed changes (slowing down).
2. Cluster these points using k-means or other methods
3. Create one node for each cluster, and label it as an intersection.
4. Create links between intersections from trajectory data that hits
multiple intersections.
"""                  