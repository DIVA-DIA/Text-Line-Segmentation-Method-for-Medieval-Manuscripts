import logging
import time


def polygon_to_string(polygons):
    # -------------------------------
    start = time.time()
    # -------------------------------
    strings = []
    for polygon in polygons:
        line_string = []
        for i, point in enumerate(polygon[0]):
            if i % 3 != 0:
                continue
            line_string.append("{},{}".format(int(point[1]), int(point[0])))
        strings.append(' '.join(line_string))

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------
    return strings