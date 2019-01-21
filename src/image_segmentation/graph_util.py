import networkx as nx
import numpy as np
import scipy.spatial
import cv2
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

    TIN = scipy.spatial.Delaunay(points)
    edges = set()
    # for each Delaunay triangle
    for n in range(TIN.nsimplex):
        # for each edge of the triangle
        # sort the vertices
        # (sorting avoids duplicated edges being added to the set)
        # and add to the edges set
        edge = sorted([TIN.vertices[n, 0], TIN.vertices[n, 1]])
        edges.add((edge[0], edge[1]))
        edge = sorted([TIN.vertices[n, 0], TIN.vertices[n, 2]])
        edges.add((edge[0], edge[1]))
        edge = sorted([TIN.vertices[n, 1], TIN.vertices[n, 2]])
        edges.add((edge[0], edge[1]))

    # make a graph based on the Delaunay triangulation edges
    graph = nx.Graph(list(edges))

    original_nodes = points
    for n in range(len(original_nodes)):
        XY = original_nodes[n]  # X and Y tuple - coordinates of the original points
        graph.node[n]['XY'] = [XY[1], XY[0]]

    return graph


def print_graph_on_img(img, graph):
    for edge in graph.edges:
        p1 = np.asarray(graph.nodes[edge[0]]['XY'], dtype=np.uint32)
        p1 = (p1[0], p1[1])
        p2 = np.asarray(graph.nodes[edge[1]]['XY'], dtype=np.uint32)
        p2 = (p2[0], p2[1])
        cv2.line(img, p1, p2, (0, 255, 0), thickness=3)


def cut_graph_with_seams(graph, seams):
    edges_to_remove = set()

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
                edges_to_remove.add(edge)

    graph.remove_edges_from(edges_to_remove)


def graph_to_point_lists(graph):
    if nx.is_connected(graph):
        return [graph._node[node]['XY'] for node in graph._node]

    graphs = np.asarray([g for g in nx.connected_component_subgraphs(graph)])
    return [[g._node[node]['XY'] for node in g._node] for g in detect_small_graphs(graphs)]


def detect_small_graphs(graphs):
    copy = [np.asarray(list(g.nodes)) for g in graphs]
    graph_sizes = np.asarray([g.size for g in copy])
    # threshold which graphs to discard
    too_small = abs(graph_sizes - np.mean(graph_sizes)) < np.std(graph_sizes)
    return graphs[too_small]
