"""
"""

# Utils
import argparse
import logging

import cv2
import numpy as np
from XMLwriter import writePAGEfile
from scipy.spatial import ConvexHull
from skimage import measure
from sklearn.cluster import DBSCAN

#######################################################################################################################
# Argument Parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Textline Segmentation')

# Inputs
parser.add_argument('--segmented_image', help='path of the segmented image')

# Parse
args = parser.parse_args()


#######################################################################################################################

def segment_textlines(input_loc, output_loc, eps=0.01, min_samples=5, simplified=True, visualize=False):
    """
    Function to compute the text lines from a segmented image. This is the main routine where the magic happens
    :param input_loc: path to segmented image
    :param output_loc: path to save generated PAGE XML
    :param a: param_a
    :param b: param_b
    """
    #############################################
    # Load the image
    img = cv2.imread(input_loc)

    # Prepare image (filter only text, ...)
    img = prepare_image(img)

    # Show the image to screen
    if visualize:
        cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
        cv2.resizeWindow('image', 1200, 800)
        cv2.moveWindow('image', 200, 100)

    #############################################
    # Find CC
    cc_labels = measure.label(img, background=0)
    cc_properties = measure.regionprops(cc_labels, cache=True)

    # Collect CC centroids
    all_centroids = []
    for cc in cc_properties:
        all_centroids.append(cc.centroid[0:2])
    all_centroids = np.asarray(all_centroids)

    # Collect CC sizes
    area = []
    for cc in cc_properties:
        area.append(cc.area)

    # Draw centroids [ALL]
    for c in all_centroids:
        # On their location
        cv2.circle(img, tuple(reversed(np.round(c).astype(np.int))), radius=5,
                   color=(200, 0, 0), thickness=10, lineType=1, shift=0)
        # On the side
        cv2.circle(img, tuple([50, np.round(c[0]).astype(np.int)]), radius=2,
                   color=(200, 0, 0), thickness=2, lineType=1, shift=0)

    #############################################
    # Discard outliers & sort
    no_outliers = detect_outliers(all_centroids[:, 0], area)
    centroids = all_centroids[no_outliers, :]
    centroids = centroids[np.argsort(centroids[:, 0]), :]

    # Draw centroids [NO_OUTLIERS]
    for i, c in enumerate(centroids):
        # On their location
        cv2.circle(img, tuple(reversed(np.round(c).astype(np.int))), radius=5,
                   color=(0, 200, 0), thickness=10, lineType=1, shift=0)
        # On the side
        cv2.circle(img, tuple([25, np.round(c[0]).astype(np.int)]), radius=2,
                   color=(0, 200, 0), thickness=2, lineType=1, shift=0)

    #############################################
    # Cluster the points and draw the clusters
    # TODO add the area as 2nd dimensions instead of zeros?
    centroids, labels = cluster(img, centroids, eps, min_samples)
    clusters_lines = draw_clusters(img, centroids, labels)

    # Draw centroids [AFTER CLUSTERING]
    for i, c in enumerate(centroids):
        if no_outliers[i]:
            # On their location
            cv2.circle(img, tuple(reversed(np.round(c).astype(np.int))), radius=5,
                       color=(0, 0, 200), thickness=10, lineType=1, shift=0)

    # Restore the outliers removed by clustering
    no_outliers = detect_outliers(all_centroids[:, 0], area)
    centroids = all_centroids[no_outliers, :]
    centroids = centroids[np.argsort(centroids[:, 0]), :]

    # Separate the centroids in cluster bins
    clusters_centroids = [[]]
    l = 0
    for c in zip(centroids):
        if l == len(clusters_lines) or c[0][0] < clusters_lines[l]:
            clusters_centroids[l].append(c[0])
        else:
            l += 1
            clusters_centroids.append([])
            clusters_centroids[l].append(c[0])

    # Sort the bins according to the horizontal axis
    for i in range(0, len(clusters_centroids)):
        clusters_centroids[i] = np.asarray(clusters_centroids[i])
        clusters_centroids[i] = clusters_centroids[i][np.argsort(clusters_centroids[i][:, 1]), :]

    # ******************************************* *******************************************
    if simplified:
        boxes = []
        for line in clusters_centroids:
            left = np.round(np.min(line[:, 1])).astype(np.int)
            top = np.round(np.min(line[:, 0])).astype(np.int)
            right = np.round(np.max(line[:, 1])).astype(np.int)
            bottom = np.round(np.max(line[:, 0])).astype(np.int)
            boxes.append("{},{} {},{} {},{} {},{}".format(top, left, top, right, bottom, left, bottom, right))

        # Save bounding box for each row as PAGE file
        writePAGEfile(output_loc, textLines=boxes)
    # ******************************************* *******************************************
    else:
        # Create a working copy of the image to draw the CC convex hull & so
        cc_img = np.zeros(img.shape[0:2])

        # Connect the centroid inside each cluster by drawing a white line
        for line in clusters_centroids:
            for i in range(0, len(line) - 1):
                # Draw the line between the centroids
                cv2.line(img, tuple([np.round(line[i][1]).astype(np.int), np.round(line[i][0]).astype(np.int)]),
                         tuple([np.round(line[i + 1][1]).astype(np.int), np.round(line[i + 1][0]).astype(np.int)]),
                         color=(255, 127, 0), thickness=5, lineType=8, shift=0)
                # Draw it on the working copy
                cv2.line(cc_img, tuple([np.round(line[i][1]).astype(np.int), np.round(line[i][0]).astype(np.int)]),
                         tuple([np.round(line[i + 1][1]).astype(np.int), np.round(line[i + 1][0]).astype(np.int)]),
                         color=(255, 255, 255), thickness=5, lineType=8, shift=0)
                # On their location
                # cv2.circle(img, tuple([np.round(line[0][i][1]).astype(np.int),
                #                     np.round(line[0][i][0]).astype(np.int)]), radius=10,
                #           color=(255, 0, 127), thickness=10, lineType=1, shift=0)

        #############################################
        # Extract the contour of each CC
        cc_polygons = []
        l = 0
        for line in clusters_centroids:
            cc_polygons.append([])
            for c in line:
                cc = find_cc_from_centroid(c, cc_properties)
                points = cc.coords[::3, 0:2]
                hull = ConvexHull(points)
                cc_polygons[l].append(points[hull.vertices][:, [1, 0]])
                # for v in hull.vertices:
                #     cv2.circle(img, tuple(reversed(points[v])), radius=1, color=(0, 255, 255), thickness=5)
                cv2.polylines(img, [points[hull.vertices][:, [1, 0]]], isClosed=True, thickness=5, color=(0, 255, 255))
                # Draw the convex hulls of each CC on the working copy
                cv2.fillPoly(cc_img, [points[hull.vertices][:, [1, 0]]], color=(255, 255, 255))
                #cv2.fillPoly(img, [points[hull.vertices][:, [1, 0]]], color=(0, 255, 255))
            l += 1

        # TODO at some point it has to be implemented that BLUE CC, (too big, not the too small) has to be dealt with.
        # TODO the plan is that if the part "invading" another ling DO NOT CROSS the bluish line connecting the centroid,
        # TODO then the whole CC belongs to such line and we will convex_hull it. OTHERWISE we crop this component on the
        # TODO median between the two lines (green line separating the clusters)

        # Compute the big polygon matching all of them


        #cv2.polylines(img, [boundary.hull.points[boundary.hull.vertices][:, [1, 0]]], isClosed=True, thickness=5, color=(0, 255, 255))

    if visualize:
        # Print again
        cv2.imshow('image', img)

        # Hold on
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print workig copy
        #cv2.imshow('image', cc_img)

        # Hold on
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

#######################################################################################################################


def prepare_image(img):
    # img = img[:,:,1] #TODO make it select only the text pixels from SegImg
    img[:, :, 0] = 0
    img[:, :, 2] = 0
    locations = np.where(img == 127)
    img[:, :, 1] = 0
    img[locations[0], locations[1]] = 255
    return img


def detect_outliers(centroids, area):
    big_enough = area > 0.3 * np.mean(area)
    small_enough = area < 2 * np.mean(area)
    no_y = abs(centroids - np.mean(centroids)) < 2 * np.std(centroids)
    no_outliers = [x & y & z for (x, y, z) in zip(big_enough, small_enough, no_y)]
    return no_outliers


def cluster(img, centroids, eps, min_samples):
    # Attempt clustering with DBSCAN
    X = centroids[:, 0]
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    tmp = np.zeros((X.shape[0], 2))
    tmp[:, 0] = X
    X = tmp

    #eps = 0.01  # centroids test1&2&3&4 (min sample 5) GT=4,16,13,29

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    # Draw DBSCAN outliers
    for i in range(0, len(centroids)):
        # Draw outliers
        if db.labels_[i] == -1:
            cv2.circle(img, tuple([75, np.round(centroids[i, 0]).astype(np.int)]), radius=3,
                       color=(0, 0, 127), thickness=3, lineType=1, shift=0)
            # print('{}'.format(i))

    # Remove outliers
    centroids = centroids[db.labels_ != -1, :]
    labels = db.labels_[db.labels_ != -1]
    return centroids, labels


def draw_clusters(img, centroids, labels):
    # Draw and count the cluster
    n_clusters = 1
    clusters_lines = []
    for i in range(0, len(centroids) - 1):
        # Draw the line between the clusters
        if labels[i] != labels[i + 1]:
            # Count the cluster
            n_clusters += 1
            # Compute line location
            clusters_lines.append((centroids[i, 0] + centroids[i + 1, 0]) / 2.0)
            # Draw line
            cv2.line(img, tuple([0, np.round(clusters_lines[n_clusters - 2]).astype(np.int)]),
                     tuple([4000, np.round(clusters_lines[n_clusters - 2]).astype(np.int)]),
                     color=(0, 127, 0), thickness=4, lineType=8, shift=0)
    print("C:{}".format(n_clusters))
    return clusters_lines


def find_cc_from_centroid(c, cc_properties):
    for cc in cc_properties:
        if (np.asarray(cc.centroid[0:2]) == c).all():
            return cc
    return None


#######################################################################################################################

if __name__ == "__main__":
    # Set up logging to console
    formatter = logging.Formatter(
        fmt='%(asctime)s %(funcName)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    stderr_handler = logging.StreamHandler()
    stderr_handler.formatter = formatter
    logging.getLogger().addHandler(stderr_handler)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s',
        level=logging.INFO)
    logging.info('Printing activity to the console')
    segment_textlines(input_loc='./../data/test1.png', output_loc="./../data/testfile.txt")

    ################################################################################
    ################################################################################
    ################################################################################
    """
    from __future__ import print_function

    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Generating the sample data from make_blobs
    # This particular setting has one distinct cluster and 3 clusters placed close
    # together.
    X, y = make_blobs(n_samples=500,
                      n_features=2,
                      centers=4,
                      cluster_std=1,
                      center_box=(-10.0, 10.0),
                      shuffle=True,
                      random_state=1)  # For reproducibility

    range_n_clusters = [2, 3, 4, 5, 6]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()
"""

    ################################################################################
    ################################################################################
    ################################################################################