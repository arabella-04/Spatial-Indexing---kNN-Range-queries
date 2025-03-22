import pandas as pd
import numpy as np
import math 
import heapq

# calculates euclidean distance from 2 pois
def euclidean_distance(point_x, point_y):
    return ((point_x[0]-point_y[0])**2 + (point_x[1]-point_y[1])**2)**0.5

# O(n) - technically O(nlogk)
def knn_linear_search(data, target_id, k):
    """Takes as input the dataset, a target POI ID, and the number of neighbors (k).
    – Finds the target POI and computes its distance to all other POIs.
    – Returns a list of the k-nearest neighbors and their distances."""
    # linear scan to find the target poi
    for poi in data:
        if poi[0]==target_id:
            target_poi = poi
            break
    # calculate distance from target to all pois
    dist = np.array([euclidean_distance((poi[1], poi[2]), (target_poi[1], target_poi[2]))for poi in data])
    # append distance as a column to the data
    data = np.hstack((data, dist.reshape(-1, 1)))
    data_list = data.tolist()
    # use a min-heap to find k smallest distances
    neighbors = heapq.nsmallest(k+1, data_list, key=lambda x: x[4])
    return np.array(neighbors[1:], dtype=object)

# O(n)
def range_query_linear_search(data, target_id, r):
    """Takes as input the dataset, a target POI ID, and a radius r.
    – Returns a list of POIs within the radius r from the target"""
    # linear scan to find the target poi
    for poi in data:
        if poi[0]==target_id:
            target_poi = poi
            break
    # calculate distance from target to all pois
    dist = np.array([euclidean_distance((poi[1], poi[2]), (target_poi[1], target_poi[2]))for poi in data])
    # append distance as a column to the data
    data = np.hstack((data, dist.reshape(-1, 1)))
    # a linear scan to get all the pois with distance <= r
    range = data[(data[:, 0] != target_id) & (data[:, 4] <= r)]
    return range

# O(n)
def build_grid_index(data, cell_size):
    """Takes as input the dataset of POIs and the size of each cell (cell size).
    – Assigns each POI to the appropriate grid cell based on its latitude and longitude.
    – Returns the grid index"""
    grid = {}
    max_lat = 45.0159
    max_lon = -71.8562
    min_lat = 40.4774
    min_lon = -79.7624
    # create an empty grid that divides the space into uniform `cell_size` x `cell_size` cells
    for x in range(math.floor((max_lon - min_lon) / cell_size) + 1):
        for y in range(math.floor((max_lat - min_lat) / cell_size) + 1):
            grid[(x, y)] = []
    # a linear scan through the data to map each poi to its correct cell
    for poi in data:
        x = math.floor((poi[2] - min_lon) / cell_size)
        y = math.floor((poi[1] - min_lat) / cell_size)
        grid[(x, y)].append(poi)
    return grid

# O(√N log k), worst case - O(N log k)
def knn_grid_index(grid, target_poi, k, cell_size):
    """Takes as input the dataset, a target POI, and the number of neighbors (k).
    – Finds the POIs closest to the target POI and performs a linear search amomg those POIs to find the k nearest ones.
    – Returns a list of the k-nearest neighbors and their distances."""

    max_lat = 45.0159
    max_lon = -71.8562
    min_lat = 40.4774
    min_lon = -79.7624
    # find the indices of the cell for target poi
    target_x = math.floor((target_poi[2] - min_lon) / cell_size)
    target_y = math.floor((target_poi[1] - min_lat) / cell_size)
    # a list to store all the pois close to the target
    subset = []
    subset.extend(grid[(target_x, target_y)])
    i = 1
    # Expanding outward in a concentric square pattern to collect at least k+1 nearest pois.
    while len(subset)<k+1:
        cells = []
        for dx in range(-i, i+1):
            for dy in range(-i, i+1):
                if max(abs(dx), abs(dy))==i:
                    cells.append((target_x+dx, target_y+dy))
        for cell in cells:
            if cell in grid:
                subset.extend(grid[cell])
        i+=1
    # adding 2 extra layers because a point directly outside the 
    # initial subset but straight down (or in another cardinal direction) 
    # can be closer than a diagonally included point. This ensures that all 
    # potentially closer points are considered before selecting the k nearest neighbors.
    for _ in range(2):
        cells = []
        for dx in range(-i, i+1):
            for dy in range(-i, i+1):
                if max(abs(dx), abs(dy))==i:
                    cells.append((target_x+dx, target_y+dy))
        for cell in cells:
            if cell in grid:
                subset.extend(grid[cell])
        i+=1
    # performing linear search with a samller subset of the original data
    return knn_linear_search(np.array(subset, dtype=object), target_poi[0], k)

# O(n)
def range_query_grid_index(grid, target_poi, r, cell_size):
    """Takes as input the dataset, a target POI ID, and a radius r.
    – Returns a list of POIs within the radius r from the target"""
    max_lat = 45.0159
    max_lon = -71.8562
    min_lat = 40.4774
    min_lon = -79.7624
    # find the indices of the cell for target poi
    target_x = math.floor((target_poi[2] - min_lon) / cell_size)
    target_y = math.floor((target_poi[1] - min_lat) / cell_size)
    # a list to store all the pois close to the target
    subset = []
    # number of cells to go in each direction based r & cell_size
    num_cells = math.ceil(r / cell_size)
    # collect pois from all cells that are within range from the target
    for dx in range(-num_cells, num_cells + 1):
        for dy in range(-num_cells, num_cells + 1):
            neighbor_x = target_x + dx
            neighbor_y = target_y + dy           
            if (neighbor_x, neighbor_y) in grid:
                subset.extend(grid[(neighbor_x, neighbor_y)])
    # performing linear search with a samller subset of the original data
    return range_query_linear_search(np.array(subset, dtype = object), target_poi[0], r)

# a node for KD tree
class KDNode:
    def __init__(self, point, dim):
        self.point = point
        self.dim = dim # dimension the layer is split on - lat or lon
        self.left = None
        self.right = None

#O(nlogn)
def build_kd_tree(data, depth=0):
    """Takes as input the dataset of POIs.
    – Constructs a KD-Tree from scratch using the coordinates (latitude, longitude) of the POIs.
    – Returns the root node of the constructed KD-Tree.    """
    if len(data) == 0:
        return None
    # alternate between lat and lon
    dim = depth % 2  
    sort_col = dim + 1  
    # sort the data by the chosen dim
    sorted_idx = np.argsort(data[:, sort_col].astype(float))
    sorted_data = data[sorted_idx]
    # median 
    median_idx = len(sorted_data) // 2
    # creating a node at the median
    node = KDNode(sorted_data[median_idx], dim)
    # recursively build left and right subtrees
    node.left = build_kd_tree(sorted_data[:median_idx], depth + 1)
    node.right = build_kd_tree(sorted_data[median_idx+1:], depth + 1)
    return node

def euclidean_distance_sq(p1, p2):
    return (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2

def knn_kd_tree(root, target_poi, k):
    """
    Takes as input the root of a KD-Tree, a target POI, and the number of neighbors (k).
    – Finds the POIs closest to the target POI by traversing the KD-Tree.
    – Returns a list of the k-nearest neighbors and their distances.
    """

    best = []
    k = k+1
    def add_candidate(dist_sq, candidate_point):
        # add or replace candidate in heap of size k
        if len(best) < k:
            heapq.heappush(best, (-dist_sq, candidate_point[0], candidate_point))
        else:
            worst_dist_sq = -best[0][0]
            if dist_sq < worst_dist_sq:
                heapq.heapreplace(best, (-dist_sq, candidate_point[0], candidate_point))
    def search(node):
        if node is None:
            return
        dist_sq = euclidean_distance_sq(node.point, target_poi)
        add_candidate(dist_sq, node.point)
        dim = node.dim  # 0 - lat, 1 - lon
        node_coord = float(node.point[dim+1])
        target_coord = float(target_poi[dim+1])
        # choose primary branch
        if target_coord < node_coord:
            search(node.left)
        else:
            search(node.right)
        # worst distance from heap
        worst_dist_sq = -best[0][0] if best else float('inf')
        axis_diff_sq = (target_coord - node_coord)**2
        # check if opposite branch is needed
        if (len(best) < k) or (axis_diff_sq <= worst_dist_sq):
            if target_coord < node_coord:
                search(node.right)
            else:
                search(node.left)
    search(root)
    results = []
    # pop elements from heap
    while best:
        neg_sq, _, pt = heapq.heappop(best)
        # compute euclidean distance
        dist = (-neg_sq)**0.5
        results.append([pt[0], pt[1], pt[2], pt[3], dist])
    # reverse for nearest-first
    results.reverse()
    return np.array(results[1:], dtype=object)

# O(logn)
def range_query_kd_tree(root, target_poi, r):
    """Finds all POIs within a radius r from the target POI using a KD-Tree search."""
    results = []
    stack = [root]
    tid = target_poi[0]
    tx = target_poi[1] 
    ty = target_poi[2]  
    # traverses the KD-Tree to find all POIs within the given radius r.
    while stack:
        node = stack.pop()
        if node is None:
            continue
        nx = node.point[1]
        ny = node.point[2]
        dist = euclidean_distance((tx, ty), (nx, ny))
        # store poi if within radius from the target
        if dist <= r and node.point[0] != tid:
            results.append([node.point[0], nx, ny, node.point[3], dist])
        # determine which branch to explore
        axis = node.dim
        node_coord = nx if axis == 0 else ny
        target_coord = tx if axis == 0 else ty
        diff = target_coord - node_coord
        # primary branch
        if diff < 0:
            stack.append(node.left)
        else:
            stack.append(node.right)
        # opposite branch if within r
        if abs(diff) <= r:
            if diff < 0:
                stack.append(node.right)
            else:
                stack.append(node.left)
    return np.array(results, dtype=object)

if __name__ == '__main__':
    from plots import plot_knn_linear, plot_range_query_linear
    from plots import plot_grid_knn, plot_grid_range_query, plot_grid_knn_cell_size, plot_grid_range_query_cell_size
    from plots import plot_grid_vs_linear_knn, plot_grid_vs_linear_range_query, plot_build_time
    from plots import plot_kd_knn, plot_kd_range_query
    from plots import plot_kd_vs_linear_knn, plot_kd_vs_linear_range_query, plot_kd_vs_grid_knn, plot_kd_vs_grid_range_query

    df = pd.read_csv('interpreter.csv')
    print(df.shape)
    # Removing entries with missing or malformed data
    df = df.dropna(subset=['@id', '@lon', '@lat'])
    # Ensuring that the unique identifier (@id) consists only of numeric characters.
    df = df.loc[df['@id'].astype('str').str.isdigit()]
    # Ensuring that the unique identifier (@id) is unique to each POI.
    df = df.drop_duplicates(subset='@id')
    print(df.shape)

    N = [1000, 10000, 100000, 825171, 1617988]
    K = [1, 5, 10, 50, 100, 500]
    R = [0.01, 0.05, 0.1, 0.2, 0.5]
    cell_sizes = [0.01, 0.05, 0.1, 0.2]

    plot_knn_linear(df, N, K, num_trials=10)
    plot_range_query_linear(df, N, R, num_trials=10)
    plot_grid_knn(df, N, K, num_trials= 10, cell_size = 0.01)
    plot_grid_range_query(df, N, R, num_trials=10 , cell_size = 0.05)
    plot_grid_knn_cell_size(df, N, cell_sizes, num_trials=10, k = 5)
    plot_grid_range_query_cell_size(df, N, cell_sizes, num_trials = 10, r = 0.1)
    plot_grid_vs_linear_knn(df, N, num_trials = 10, k = 5, cell_size = 0.05)
    plot_grid_vs_linear_range_query(df, N, num_trials = 10, r = 0.1, cell_size = 0.05)
    plot_kd_knn(df, N, K, num_trials=10)
    plot_kd_vs_linear_knn(df, N, num_trials = 10, k = 5)
    plot_kd_vs_grid_knn(df, N, num_trials = 10, k = 5, cell_size = 0.05)
    plot_build_time(df, N, cell_sizes, num_trials=10)
    plot_kd_range_query(df, N, R, num_trials=10)
    plot_kd_vs_linear_range_query(df, N, num_trials=10, r = 0.1)
    plot_kd_vs_grid_range_query(df, N, num_trials = 10, r = 0.5, cell_size = 0.05)
