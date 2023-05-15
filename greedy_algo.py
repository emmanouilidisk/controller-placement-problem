import numpy as np
import networkx as nx
import random
import itertools

def maxindex(dist, n):
    mi = 0
    for i in range(n):
        if (dist[i] > dist[mi]):
            mi = i
    return mi

def k_center(graph, k):
    """
    Implementation of the k-center algorithm.
    ------
    Inputs:
    - graph: networkx graph object
    - k: number of centers

    Ouputs:
    - list with centers
    - maximum distance between a vertex and centers
    """

    n = graph.number_of_nodes()
    dist = [0] * n
    centers = []
    weights = nx.get_edge_attributes(graph, "LinkSpeed")
    for i in range(n):
        dist[i] = 10 ** 9

    # index of city having the
    # maximum distance to it's
    # closest center
    max_dist = 0
    for i in range(k):
        centers.append(max_dist)
        for j in range(n):
            # updating the distance
            # of the cities to their
            # closest centers
            temp = (str(max_dist),str(j))
            if temp in weights.keys():
                dist[j] = min(dist[j], float(weights[temp]))
            else:
                continue

        # updating the index of the
        # city with the maximum
        # distance to it's closest center
        max_dist = maxindex(dist, n)
    return (centers, max_dist)


def k_star_means(G, K, k_star, theta=0.5, flag=0):
    # Step 1: Initialize K* clusters
    centers = random.sample(list(G.nodes()), k_star)
    if(flag):
        centers[0] = 0
    clusters = [[] for i in range(k_star)]
    adjacency_matrix = [[0 for _ in range(n)] for _ in range(n)]
    edges = nx.get_edge_attributes(G, "LinkSpeed")
    for elem in edges:
        adjacency_matrix[int(elem[0])][int(elem[1])] = float(edges[elem])
        adjacency_matrix[int(elem[1])][int(elem[0])] = float(edges[elem])
    G = nx.DiGraph(np.array(adjacency_matrix))


    # Step 2: Assign nodes to nearest cluster
    for node in G.nodes():
        distances = [float(adjacency_matrix[int(node)][int(center)]) for center in centers]
        nearest_center_index = distances.index(min(distances))
        clusters[nearest_center_index].append(int(node))
    flag1 = False

    # Step 3-7: Iterate until K clusters are obtained
    while len(clusters) > K:
        # Step 3: Update cluster centers
        centers = []
        for cluster in clusters:
            sum_distances = {int(node): 0 for node in G.nodes()}
            for node in cluster:
                for center in centers:
                    sum_distances[center] += nx.shortest_path_length(G, int(node), center, weight='LinkSpeed')
            center = min(sum_distances, key=sum_distances.get)
            centers.append(center)

        # Step 4: Calculate function values for pairs of clusters
        function_values = {}
        d = np.zeros((len(clusters), len(clusters)))
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist_list = []
                cluster_i = set(clusters[i])
                cluster_j = set(clusters[j])
                if(len(cluster_i)<1 or len(cluster_j)<1):
                    
                    flag1 = True
                    break
                for node_s, node_t in itertools.product(cluster_i, cluster_j):
                    if(nx.has_path(G, node_s, node_t)):
                        dist_list.append(nx.shortest_path_length(G, source=int(node_s), target=int(node_t), weight='LinkSpeed'))
                d[i][j] = min(dist_list)
            if(flag1):
                break

        # print(dist_list)
        sum_d = sum(sum(d))
        if(sum_d == 0):
            print(sum_d)
        K_c = len(clusters)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_i = set(clusters[i])
                cluster_j = set(clusters[j])
                function_values[(i, j)] = theta * (K_c *d[i][j] / sum_d) + (1-theta) * k *(len(cluster_i)+len(cluster_j)) / n

        # Step 5: Find smallest function value
        min_value = min(function_values.values())
        min_clusters = [pair for pair in function_values if function_values[pair] == min_value][0]

        # Step 6: Merge smallest pair of clusters
        new_cluster = clusters[min_clusters[0]] + clusters[min_clusters[1]]
        del clusters[min_clusters[1]]
        clusters[min_clusters[0]] = new_cluster
    centers = []
    for cluster in clusters:
        sum_distances = {int(node): 0 for node in G.nodes()}
        for node in cluster:
            for center in centers:
                sum_distances[center] += nx.shortest_path_length(G, int(node), center, weight='LinkSpeed')
        center = min(sum_distances, key=sum_distances.get)
        centers.append(center)

    max_dist = compute_max_dist(adjacency_matrix, centers)
    return (centers, max_dist)

def compute_max_dist(adjacency_matrix, centers):
    """
    Function Computing the maximum distance between from nodes to the closest center.

    Inputs:
    adjacency_matrix (list): Adjacency matrix of the graph
    centers (list): centers of the topology

    Returns:
        max_dist: the maximum distance from nodes to closest centers
    """
    max_dist = 0
    for i in range(len(centers)):
        center = centers[i]
        for row in adjacency_matrix:
            if(row[center] > max_dist):
                max_dist = row[center]
    return max_dist

def k_means(X: np.ndarray, k: int, max_iters: int = 1000):
    """
    K-means clustering algorithm implementation.

    Args:
    X (np.ndarray): Input data matrix of shape (n_samples, n_features).
    k (int): Number of clusters to create.
    max_iters (int): Maximum number of iterations for convergence.

    Returns:
    List[int]: A list of integers representing the cluster assignments for each input sample.
    """

    # Randomly initialize k centroids from input data.
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    # Initialize cluster assignments.
    cluster_assignments = np.zeros(X.shape[0])

    # Main k-means loop.
    for _ in range(max_iters):
        # Assign each data point to the closest centroid.
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        new_assignments = np.argmin(distances, axis=1)

        # Check for convergence.
        if np.array_equal(cluster_assignments, new_assignments):
            break

        # Update cluster assignments and centroids.
        cluster_assignments = new_assignments
        for i in range(k):
            cluster_points = X[cluster_assignments == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)

    return centroids

def k_self_adaptive(G):
    """
    Implementation of k-self Adaptive algorithm from:

    Input:
    G (networkX object): the graph of the network
    Output:
    centers (list): the centers of the topology
    max_dist (float): the maximum distance from a node to the closest center
    """
    # variable initialization
    n = nx.number_of_nodes(G)
    min_eig_gap = 1e8
    adjacency_matrix = [[0 for _ in range(n)] for _ in range(n)]
    edges = nx.get_edge_attributes(G, "LinkSpeed")
    V_k = []

    # find the number of controllers k
    for elem in edges:
        adjacency_matrix[int(elem[0])][int(elem[1])] = float(edges[elem])
        adjacency_matrix[int(elem[1])][int(elem[0])] = float(edges[elem])
    eigenvalues, eigenvectors = np.linalg.eig(adjacency_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvalues.sort()
    for i in range(n-1):
        g_i = eigenvalues[i+1] - eigenvalues[i]
        if(eigenvalues[i+1] == eigenvalues[i]):
            continue
        if (g_i < min_eig_gap):
            index = i
            min_eig_gap = g_i
    k = int(np.ceil(min_eig_gap))

    # compute graph Laplacian L and matrix V from paper
    L = np.negative(np.array(adjacency_matrix))
    for i in range(len(graph)):
        L[i][i] = sum(graph[i])
    L_eigenvalues, V = np.linalg.eig(L)
    idx = L_eigenvalues.argsort()[::-1]
    for index in idx[:k]:
        V_k.append(V[:, index])
    V_k = np.array(V_k)
    V_k = V_k.real

    # run k-means for the rows in V
    centers = np.argmax(k_means(V_k, k), axis=1)

    # compute max distance from node to closest center
    max_dist = compute_max_dist(adjacency_matrix, centers)
    return(centers, max_dist)
def is_connected(adj_matrix):
    """
    Check if a graph represented by an adjacency matrix is connected.
    """
    # Get the number of nodes in the graph
    num_nodes = len(adj_matrix)

    # Create a set to keep track of visited nodes
    visited = set()

    # Use a stack to keep track of nodes to visit
    stack = [0]

    # Visit nodes until there are no more nodes to visit
    while stack:
        # Pop the top node from the stack
        node = stack.pop()

        # Mark the node as visited
        visited.add(node)

        # Get the neighbors of the node
        neighbors = set(i for i in range(num_nodes) if adj_matrix[node][i] != 0)

        # Add any unvisited neighbors to the stack
        stack.extend(neighbors - visited)

    # If all nodes were visited, the graph is connected
    return len(visited) == num_nodes
if __name__ == "__main__":
    mygraph = [[5*i for i in range(25)]]
    # G = np.array(mygraph)
    graph = np.array([[0, 1, 2, 3],
                  [1, 0, 1, 2],
                  [2, 1, 0, 1],
                  [3, 2, 1, 0]])
    k = 1
    # read gfile
    G = nx.read_graphml('a+.graphml')
    n = nx. number_of_nodes(G)
    for i in range(100):
        # run k-center algorithm
        centers1, max_dist1 = k_center(G, k)
        print(centers1, max_dist1)

        # run k-Self Adaptive Algorithm
        centers2, max_dist2 = k_self_adaptive(G)
        print(centers2, max_dist2)

        # run k_star Algorithm
        k_star = 2*k
        centers3, max_dist3 = k_star_means(G, k, k_star, theta=1)
        print(centers3, max_dist3)