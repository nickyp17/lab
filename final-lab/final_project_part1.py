import min_heap
import random
import matplotlib.pyplot as plt
import numpy as np
import timeit

class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)


def dijkstra(G, source):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    #Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(source, 0)

    #Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
    return dist


def dijkstra_approx(G, source, k):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    relax = {} #Relaxation dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    #Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
        relax[node] = 0
    Q.decrease_key(source, 0)

    #Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour] and relax[current_node] < k:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
                relax[neighbour] = relax[current_node] + 1
    return dist



def bellman_ford(G, source):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    nodes = list(G.adj.keys())

    #Initialize distances
    for node in nodes:
        dist[node] = float("inf")
    dist[source] = 0

    #Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour):
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist


def bellman_ford_approx(G, source, k):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    relax = {} #Relaxation dictionary
    nodes = list(G.adj.keys())

    #Initialize distances
    for node in nodes:
        dist[node] = float("inf")
        relax[node] = 0
    dist[source] = 0

    #Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour) and relax[node] < k:
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
                    relax[neighbour] = relax[node] + 1
    return dist


def total_dist(dist):
    total = 0
    for key in dist.keys():
        total += dist[key]
    return total


# Experiment Suite 1
def time_to_graph_size_experiment(max_n, k=2):
    '''
    Test the running time of Dijkstra's algorithm and Bellman-Ford's algorithm, as well as their respecitve approximations
    compared to the size of the graph.
    :param max_n:
    :return:
    '''

    dijkstra_times = []
    dijkstra_approx_times = []
    bellman_ford_times = []
    bellman_ford_approx_times = []

    for n in range(1, max_n + 1):
        G = create_random_complete_graph(n, 10)
        source = 0

        dijkstra_time = 0
        dijkstra_approx_time = 0
        bellman_ford_time = 0
        bellman_ford_approx_time = 0

        for _ in range(10):
            dijkstra_time += time_dijkstra(G, source)
            dijkstra_approx_time += time_dijkstra_approx(G, source, k)
            bellman_ford_time += time_bellman_ford(G, source)
            bellman_ford_approx_time += time_bellman_ford_approx(G, source, k)

        dijkstra_times.append(dijkstra_time / 10)
        dijkstra_approx_times.append(dijkstra_approx_time / 10)
        bellman_ford_times.append(bellman_ford_time / 10)
        bellman_ford_approx_times.append(bellman_ford_approx_time / 10)

    plt.plot(range(1, max_n + 1), dijkstra_times, label="Dijkstra")
    plt.plot(range(1, max_n + 1), dijkstra_approx_times, label="Dijkstra Approximation")
    plt.plot(range(1, max_n + 1), bellman_ford_times, label="Bellman-Ford")
    plt.plot(range(1, max_n + 1), bellman_ford_approx_times, label="Bellman-Ford Approximation")
    plt.xlabel("Number of nodes")
    plt.ylabel(f"Running time (seconds) (k={k})")
    plt.title("Shortest Path Running Time vs. Number of Nodes")
    plt.legend()
    plt.show()


def accuracy_to_graph_size_experiment(max_nodes, k=2):
    '''
    Test the accuracy of Dijkstra's algorithm approximation and Bellman-Ford's algorithm approximation to the size of the graph.
    :param max_nodes:
    :return:
    '''

    dijkstra_approx_errors = []
    bellman_ford_approx_errors = []

    for n in range(1, max_nodes + 1):
        G = create_random_complete_graph(n, 10)
        source = 0

        dijkstra_approx_error = 0
        bellman_ford_approx_error = 0

        for _ in range(10):
            dijkstra_approx_error += abs(total_dist(dijkstra(G, source)) - total_dist(dijkstra_approx(G, source, k)))
            bellman_ford_approx_error += abs(total_dist(bellman_ford(G, source)) - total_dist(bellman_ford_approx(G, source, k)))

        dijkstra_approx_errors.append(dijkstra_approx_error / 10)
        bellman_ford_approx_errors.append(bellman_ford_approx_error / 10)

    plt.plot(range(1, max_nodes + 1), dijkstra_approx_errors, label="Dijkstra Approximation")
    plt.plot(range(1, max_nodes + 1), bellman_ford_approx_errors, label="Bellman-Ford Approximation")
    plt.title("Shortest Path Approximation Error vs. Number of Nodes")
    plt.xlabel("Number of nodes")
    plt.ylabel(f"Error (k = {k})")
    plt.legend()
    plt.show()


def accuracy_to_num_relax_experiment(max_k):
    '''
    Test the accuracy of Dijkstra's algorithm approximation and Bellman-Ford's algorithm approximation to the number of
    relaxations allowed.
    :param max_k:
    :return:
    '''

    dijkstra_approx_errors = []
    bellman_ford_approx_errors = []

    for k in range(1, max_k + 1):
        dijkstra_approx_error = 0
        bellman_ford_approx_error = 0

        for _ in range(10):
            G = create_random_complete_graph(100, 10)
            source = 0
            dijkstra_approx_error += abs(total_dist(dijkstra(G, source)) - total_dist(dijkstra_approx(G, source, k)))
            bellman_ford_approx_error += abs(total_dist(bellman_ford(G, source)) - total_dist(bellman_ford_approx(G, source, k)))

        dijkstra_approx_errors.append(dijkstra_approx_error / 10)
        bellman_ford_approx_errors.append(bellman_ford_approx_error / 10)

    plt.plot(range(1, max_k + 1), dijkstra_approx_errors, label="Dijkstra Approximation")
    plt.plot(range(1, max_k + 1), bellman_ford_approx_errors, label="Bellman-Ford Approximation")
    plt.xlabel("Number of relaxations")
    plt.ylabel("Error (n = 100)")
    plt.title("Shortest Path Approximation Error vs. Number of Relaxations")
    plt.legend()
    plt.show()


def time_to_num_relax_experiment(max_k):
    '''
    Test the running time of Dijkstra's algorithm approximation and Bellman-Ford's algorithm approximation to the number of
    relaxations allowed.
    :param max_k:
    :return:
    '''

    dijkstra_approx_times = []
    bellman_ford_approx_times = []

    for k in range(1, max_k + 1):
        G = create_random_complete_graph(100, 10)
        source = 0

        dijkstra_approx_time = 0
        bellman_ford_approx_time = 0

        for _ in range(10):
            dijkstra_approx_time += time_dijkstra_approx(G, source, k)
            bellman_ford_approx_time += time_bellman_ford_approx(G, source, k)

        dijkstra_approx_times.append(dijkstra_approx_time / 10)
        bellman_ford_approx_times.append(bellman_ford_approx_time / 10)

    plt.plot(range(1, max_k + 1), dijkstra_approx_times, label="Dijkstra Approximation")
    plt.plot(range(1, max_k + 1), bellman_ford_approx_times, label="Bellman-Ford Approximation")
    plt.xlabel("Number of relaxations")
    plt.ylabel("Running time (seconds) (n = 100)")
    plt.title("Shortest Path Running Time vs. Number of Relaxations")
    plt.legend()
    plt.figure(figsize=(10, 10))
    plt.show()


def time_dijkstra(G, source):
    start = timeit.default_timer()
    dijkstra(G, source)
    end = timeit.default_timer()
    return end - start


def time_dijkstra_approx(G, source, k):
    start = timeit.default_timer()
    dijkstra_approx(G, source, k)
    end = timeit.default_timer()
    return end - start


def time_bellman_ford(G, source):
    start = timeit.default_timer()
    bellman_ford(G, source)
    end = timeit.default_timer()
    return end - start


def time_bellman_ford_approx(G, source, k):
    start = timeit.default_timer()
    bellman_ford_approx(G, source, k)
    end = timeit.default_timer()
    return end - start


def create_random_complete_graph(n,upper):
    G = DirectedWeightedGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i,j,random.randint(1,upper))
    return G


#Assumes G represents its nodes as integers 0,1,...,(n-1)
def mystery(G):
    n = G.number_of_nodes()
    d = init_d(G)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][j] > d[i][k] + d[k][j]: 
                    d[i][j] = d[i][k] + d[k][j]
    return d

def init_d(G):
    n = G.number_of_nodes()
    d = [[float("inf") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if G.are_connected(i, j):
                d[i][j] = G.w(i, j)
        d[i][i] = 0
    return d


def test_mystery_complexity():
    mystery_times = []

    for n in range(1, 101):
        G = create_random_complete_graph(n, 10)
        mystery_time = 0

        start = timeit.default_timer()
        mystery(G)
        end = timeit.default_timer()
        mystery_time += end - start

        mystery_times.append(mystery_time)

    plt.plot(range(1, 101), mystery_times)
    plt.xlabel("Number of nodes")
    plt.ylabel("Running time (seconds)")
    plt.title("Running time of mystery function vs. Number of nodes")
    plt.show()

