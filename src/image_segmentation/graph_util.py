import bisect
import logging
import os
import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.spatial import Delaunay
from shapely.geometry import LineString


def createTINgraph(points):
    """
    http://ssrebelious.blogspot.com/2014/11/how-to-create-delauney-triangulation.html

    Creates a graph based on Delaney triangulation

    @param points: list of points
    @return - a graph made from a Delauney triangulation

    @Copyright notice: this code is an improved (by Yury V. Ryabov, 2014, riabovvv@gmail.com) version of
                      Tom's code taken from this discussion
                      https://groups.google.com/forum/#!topic/networkx-discuss/D7fMmuzVBAw
    """
    # -------------------------------
    start = time.time()
    # -------------------------------

    TIN = Delaunay(points)
    edges = set()
    # for each Delaunay triangle
    for n in range(TIN.nsimplex):
        # for each edge of the triangle
        # sort the vertices
        # (sorting avoids duplicated edges being added to the set)
        # and add to the edges set
        edge = sorted([TIN.vertices[n, 0], TIN.vertices[n, 1]])
        # TODO weighted eucleadean distance measure
        edges.add((edge[0], edge[1], asymetric_distance(edge, points)))
        edge = sorted([TIN.vertices[n, 0], TIN.vertices[n, 2]])
        edges.add((edge[0], edge[1], asymetric_distance(edge, points)))
        edge = sorted([TIN.vertices[n, 1], TIN.vertices[n, 2]])
        edges.add((edge[0], edge[1], asymetric_distance(edge, points)))

    # make a graph based on the Delaunay triangulation edges
    graph = nx.Graph()
    graph.add_weighted_edges_from(list(edges))

    original_nodes = points
    for n in range(len(original_nodes)):
        XY = original_nodes[n]  # X and Y tuple - coordinates of the original points
        graph.node[n]['XY'] = [XY[1], XY[0]]

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return graph


def asymetric_distance(edge, points):
    return np.linalg.norm(
        np.asarray(points[edge[0]]) * np.array([1, 3]) - np.asarray(points[edge[1]]) * np.array([1, 3]))


def print_graph_on_img(img, graphs, color=(0, 255, 0), thickness=3):
    img = img.copy()
    for graph in graphs:
        for edge in graph.edges:
            p1, p2 = get_edge_node_coordinates(edge, graph)
            cv2.line(img, p1, p2, color, thickness=thickness)

    return img


def get_edge_node_coordinates(edge, graph):
    p1 = np.asarray(graph.nodes[edge[0]]['XY'], dtype=np.uint32)
    p1 = (p1[0], p1[1])
    p2 = np.asarray(graph.nodes[edge[1]]['XY'], dtype=np.uint32)
    p2 = (p2[0], p2[1])
    return p1, p2


def cut_graph_with_seams(graph, seams, nb_of_lives, too_small_pc, root_output_path):
    # -------------------------------
    start = time.time()
    # -------------------------------

    tic = time.time()
    unique_edges, weights, occurrences = find_intersected_edges(graph, seams)
    print(time.time()-tic)

    # create histogram and save it
    # plt.hist(occurrences, bins='auto')
    # plt.savefig(os.path.join(output_path, 'histo/histogram_of_intersection_occurrences.png'))

    # remove the edges from the graph
    graph.remove_edges_from(unique_edges)

    if nx.is_connected(graph):
        return list([graph])

    # get the graphs
    graphs = np.asarray(list(nx.connected_component_subgraphs(graph)))
    # detect the small ones
    small_graphs = detect_small_graphs(graphs, too_small_pc).tolist()

    # check if there are small graphs
    while small_graphs:
        # merge the small graphs
        graph = merge_small_graphs(graph, list(small_graphs), unique_edges, weights)
        # get the graphs
        graphs = np.asarray(list(nx.connected_component_subgraphs(graph)))
        # detect the small ones
        small_graphs = detect_small_graphs(graphs, too_small_pc)

    # check again if the graph is still connected
    if nx.is_connected(graph):
        return list([graph])

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return graphs


def get_edge_node_coordinates(edge, graph):
    p1 = np.asarray(graph.nodes[edge[0]]['XY'], dtype=np.uint32)
    p1 = (p1[0], p1[1])
    p2 = np.asarray(graph.nodes[edge[1]]['XY'], dtype=np.uint32)
    p2 = (p2[0], p2[1])
    return p1, p2


def get_neighbouring_seams_index(seams_max_y, seams_min_y, edge_max_y, edge_min_y):
    """
    Get the indexes of the seams which could possibly intersect with the point
    """
    # reads as "the highest (in terms of value, don't forget top left is 0,0!) value of
    # a seam has to be higher than the lowest value of the edge, and, the lowest point
    # of the seams has to be lower than the highest of the edge. In this way there CAN
    # be an intersection, but outside this range its mathematically impossible that there is.

    # IMPORTANT: note that seams_max_y and seams_min_y MUST be sorted. Here, we don't
    # sort them because of their sorted nature. As a matter of fact, it is impossible
    # for a seam to cross another one; if they meet the merge but one will never go on "the
    # other side" because of obvious reasons but most importantly the penalty. So the lists
    # are for us already sorted.
    return [bisect.bisect_left(seams_max_y, edge_min_y), bisect.bisect_left(seams_min_y, edge_max_y)]


def chunks(l, n):
    """Yield successive len(l)/n-sized chunks from l."""
    # ensure we dont ask for too many chunks
    n = np.min([n, len(l)])
    # get the size of each chunk
    size  = int(np.ceil(len(l)/n))
    for i in range(0, len(l), size):
        yield l[i:i + size]


def find_intersected_edges(graph, seams):
    # strip seams of x coordinate, which is totally useless as the x coordinate is basically the index in the array
    seams_y = [np.array(s)[:, 1] for s in seams]
    seams_max_y = np.max(seams_y, axis=1)
    seams_min_y = np.min(seams_y, axis=1)

    # node attributes (in our case the XY attribute)
    node_attributes = nx.get_node_attributes(graph, 'XY')

    # compute the lines out of seams
    seams = [LineString(seam) for seam in seams]

    # make the data iterable/splittable
    edges = [e for e in graph.edges]
    edges.sort(key=lambda tup: tup[0])

    edges_to_remove = []
    # split the data into chunks
    for chunk in chunks(edges, 250):
        # select the max/min of the edges in this chunk
        tmp = np.array(chunk)
        p1 = np.array([node_attributes[edge[0]] for edge in tmp])
        p2 = np.array([node_attributes[edge[1]] for edge in tmp])
        edge_max_y = np.max((p1[:, 1], p2[:, 1]))
        edge_min_y = np.min((p1[:, 1], p2[:, 1]))
        # get the possible seams to intersect
        index = get_neighbouring_seams_index(seams_max_y, seams_min_y, edge_max_y, edge_min_y)
        # for each edge in the chunk check if it actually does intersect
        for start, end, edge in zip(p1, p2, chunk):
            line_edge = LineString([start, end])
            for seam in seams[index[0]:index[1]]:
                if line_edge.intersects(seam):
                    edges_to_remove.append(edge)
                    break

    # # LEGACY CODE - not optimized but readable
    # edges_to_remove = []
    # for edge in graph.edges:
    #     p1 = np.asarray(node_attributes[edge[0]], dtype=np.uint32)
    #     p2 = np.asarray(node_attributes[edge[1]], dtype=np.uint32)
    #     line_edge =  LineString([p1, p2])
    #     for seam in seams:
    #         if line_edge.intersects(seam):
    #             edges_to_remove.append(edge)
    #             break

    # getting unique edges and counting them (how many times they where hit by a seam)
    unique_edges, occurrences = np.unique(np.array(edges_to_remove), return_counts=True, axis=0)
    weights = [graph.edges[edge]['weight'] for edge in unique_edges]

    return unique_edges, weights, occurrences


def merge_small_graphs(graph, small_graphs, unique_edges, weights):
    # list of edges to restore in the graph
    edges_to_add = []
    weights = np.asarray(weights)

    # TODO based on distance
    # merging small graph
    while small_graphs:
        # the indices of all edges which where cut from the given (small)graph
        small_graph = small_graphs.pop()
        edge_idxs = np.unique(np.hstack(np.asarray(
            [np.where(unique_edges == node)[0] for node in list(small_graph.nodes)])))
        # find the index in the in th edge index array
        min_edge_idx = edge_idxs[np.argmin(weights[edge_idxs])]
        # get edge to restore and add it to the list of edges to add
        edge = unique_edges[min_edge_idx]
        unique_edges = np.delete(unique_edges, min_edge_idx, axis=0)
        edges_to_add.append((edge[0], edge[1], weights[min_edge_idx]))
    # add again the edges
    graph.add_weighted_edges_from(edges_to_add)
    return graph


def graph_to_point_lists(graphs):
    return [list(nx.get_node_attributes(graph, 'XY').values()) for graph in graphs]


def detect_small_graphs(graphs, too_small_pc):
    # -------------------------------
    start = time.time()
    # -------------------------------

    graph_sizes = np.asarray([len(g.nodes) for g in graphs])
    # threshold which graphs are considered as small
    too_small = graph_sizes < too_small_pc * np.mean(graph_sizes)

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------
    # return graphs[too_small]
    return graphs[too_small]
