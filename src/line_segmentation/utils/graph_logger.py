import cv2
import os

import numpy as np

from src.line_segmentation.utils.graph_util import get_edge_node_coordinates
from src.line_segmentation.utils.util import save_img


class GraphLogger:
    IMG_SHAPE = ()
    ROOT_OUTPUT_PATH = ''

    @classmethod
    def draw_graphs(cls, img, graphs, color=(0, 255, 0), thickness=3, name='graph.png'):
        if not list(img):
            img = np.zeros(cls.IMG_SHAPE)
        else:
            img = img.copy()

        for graph in graphs:
            img = cls.draw_graph(img, graph, color, thickness, False)

        save_img(img, path=os.path.join(cls.ROOT_OUTPUT_PATH, 'graph', name), show=False)

        return img

    @classmethod
    def draw_graph(cls, img, graph, color=(0, 255, 0), thickness=3, save=False, name='graph.png'):
        if not list(img):
            img = np.zeros(cls.IMG_SHAPE)
        else:
            img = img.copy()

        cls.draw_edges(img, graph.edges, graph, color, thickness, save=False)

        if save:
            save_img(img, path=os.path.join(cls.ROOT_OUTPUT_PATH, 'graph', name), show=False)

        return img

    @classmethod
    def draw_edges(cls, img, edges, graph, color, thickness, save=False, name='graph.png'):
        for edge in edges:
            p1, p2 = get_edge_node_coordinates(edge, graph)
            cv2.line(img, p1, p2, color, thickness=thickness)

        if save:
            save_img(img, path=os.path.join(cls.ROOT_OUTPUT_PATH, 'graph', name), show=False)
