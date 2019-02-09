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
        edges.add((edge[0], edge[1], np.linalg.norm(np.asarray(points[edge[0]]) - np.asarray(points[edge[1]]))))
        edge = sorted([TIN.vertices[n, 0], TIN.vertices[n, 2]])
        edges.add((edge[0], edge[1], np.linalg.norm(np.asarray(points[edge[0]]) - np.asarray(points[edge[1]]))))
        edge = sorted([TIN.vertices[n, 1], TIN.vertices[n, 2]])
        edges.add((edge[0], edge[1], np.linalg.norm(np.asarray(points[edge[0]]) - np.asarray(points[edge[1]]))))

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


def print_graph_on_img(img, graphs, color=(0, 255, 0), thickness=3):
    img = img.copy()
    for graph in graphs:
        for edge in graph.edges:
            p1 = np.asarray(graph.nodes[edge[0]]['XY'], dtype=np.uint32)
            p1 = (p1[0], p1[1])
            p2 = np.asarray(graph.nodes[edge[1]]['XY'], dtype=np.uint32)
            p2 = (p2[0], p2[1])
            cv2.line(img, p1, p2, color, thickness=thickness)

    return img


def cut_graph_with_seams(graph, seams, nb_of_lives, root_output_path):
    # -------------------------------
    start = time.time()
    # -------------------------------

    edges_to_remove = []

    # TODO use quadtree to speed up
    for seam in seams:
        seam = LineString(seam)

        for edge in graph.edges:
            p1 = np.asarray(graph.nodes[edge[0]]['XY'], dtype=np.uint32)
            p1 = (p1[0], p1[1])
            p2 = np.asarray(graph.nodes[edge[1]]['XY'], dtype=np.uint32)
            p2 = (p2[0], p2[1])
            line_edge = LineString([p1, p2])
            if line_edge.intersects(seam):
                edges_to_remove.append(edge)

    # getting unique edges and couting them
    unique_edges, occurrences = np.unique(np.array(edges_to_remove), return_counts=True, axis=0)

    # delete the edges which get cut less then n times
    unique_edges = unique_edges[occurrences > nb_of_lives]
    # print(np.min(counter), np.max(counter))
    #plt.hist(occurrences, bins='auto')
    # plt.savefig(os.path.join(output_loc, 'histo/histo_without_reduction.png'))
    #plt.hist(occurrences[occurrences > nb_of_lives], bins='auto')
    #plt.savefig(os.path.join(root_output_path, 'histo', 'histogramm_with_reduction_and_without.png'))

    graph.remove_edges_from(unique_edges)

    if nx.is_connected(graph):
        return list([graph])

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return np.asarray(list(nx.connected_component_subgraphs(graph)))


def graph_to_point_lists(graphs):
    return [[g._node[node]['XY'] for node in g._node] for g in graphs]


def detect_small_graphs(graphs):
    # -------------------------------
    start = time.time()
    # -------------------------------

    copy = [np.asarray(list(g.nodes)) for g in graphs]
    graph_sizes = np.asarray([g.size for g in copy])
    # threshold which graphs to discard
    # too_small = abs(graph_sizes - np.mean(graph_sizes)) < 3 * np.std(graph_sizes)

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------
    # return graphs[too_small]
    return graphs[graph_sizes > 5]
