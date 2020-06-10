import math

import numpy as np
import cv2


def contour_top(contour):
    """
    Finds most top point of given contour.

    :param contour: Contour.
    :return: Top point (x, y) as numpy array.
    """
    return contour[contour[:, :, 1].argmin()][0]


def contour_bottom(contour):
    """
    Finds the extreme bottom point of given contour.

    :param contour: Contour.
    :return: Bottom point (x, y).
    """
    return contour[contour[:, :, 1].argmax()][0]


def contour_left(contour):
    """
    Finds extreme left point of given contour.

    :param contour: Contour.
    :return: Left point (x, y).
    """
    return contour[contour[:, :, 0].argmin()][0]


def contour_right(contour):
    """
    Finds extreme right point of given contour.

    :param contour: Contour.
    :return: Right point (x, y).
    """
    return contour[contour[:, :, 0].argmax()][0]


def contour_center(contour):
    """
    Calculates center vie moments.

    :param contour: Contour to calc center for.
    :return: numpy array [cx, cy].
    """
    M = cv2.moments(contour)
    cx = M['m10']/M['m00']
    cy = M['m01']/M['m00']

    return np.array([cx, cy])


def contour_area(contour):
    """
    Convenience method for cv2.contourArea.
    :param contour: Contour to calc area for.
    :return: Area.
    """
    return cv2.contourArea(contour)


def sort_contours_by_area(contours, direction=0):
    """
    Sorts contours by size ascending or descending.
    :param contours: Contours to be sorted.
    :param direction: 0 for ascending or 1 for descending sorting. Defaults to 0.
    :return: List of contours either sorted ascending (0) or descending (1).
    """
    assert direction == 0 or direction == 1, "Only direction 0 or 1 is allowed"

    if len(contours) <= 1:
        return contours

    extended_list = list(map(lambda cnt: (cnt, contour_area(cnt)), contours))

    if direction == 0:
        extended_list.sort(key=lambda tup: tup[1])  # sort by area entry asc
    else:
        extended_list.sort(key=lambda tup: -tup[1])  # sort by area entry asc

    return list(map(lambda tup: tup[0], extended_list))  # extract contours


def rectangle_corners(contour):
    """
    Assuming the given contour is a rectangle (can be rotated), this method will just return the four
    corner points.

    :param contour: Rectangle contour.
    :return: Four corner points in an order that can be drawn like a contour. As list of numpy arrays/points.
    """
    min_rect = cv2.minAreaRect(contour)

    max_width = min_rect[1][0]
    max_height = min_rect[1][1]

    # e = half of diagonal > if greater corner points will be
    # removed by Ramer Douglas Peucker algorithm
    max_epsilon = math.sqrt(max_width**2 + max_height**2) * 0.5

    # let us add some buffer
    epsilon = max_epsilon * 0.75

    corners = cv2.approxPolyDP(contour, epsilon, True)
    assert len(corners) == 4, "Somehow got not just 4 corners, got {}".format(len(corners))

    return list(map(lambda arr: arr[0], corners))
