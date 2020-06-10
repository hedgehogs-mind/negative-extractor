import math
import statistics

import cv2
import numpy as np


def points_to_line(points):
    """
    Creates line based on points. Averages lines.

    Attention!: Points are expected to be in the right order!

    Descending gradients will result in positive gradient (coordinate system is flipped y-wise).

    If the line is vertical, theta will be math.inf. In case the points go from top to bottom,
    it will be positive inf, from bottom to top > negative inf.

    :param points: List of points (x, y).
    :return: Tuple (gradient, y displacement) or (inf, x displacement) for vertical lines.
    """

    assert len(points) > 1, "Need at least two points"

    points_sorted = list(points)

    angles_sum = 0

    avg_x = statistics.mean(map(lambda p: p[0], points_sorted))
    avg_y = statistics.mean(map(lambda p: p[1], points_sorted))

    # Add all angles between two points
    for i in range(1, len(points_sorted)):
        pt1 = points_sorted[i - 1]
        pt2 = points_sorted[i]

        assert not (pt1[0] == pt2[0] and pt1[1] == pt2[1]), \
            "Got two points equal to each other, only distinct points allowed"

        delta_y = pt2[1]-pt1[1]
        length = np.linalg.norm(pt2-pt1)
        angle = delta_y/length

        angles_sum += angle

    avg_angle = angles_sum/(len(points_sorted)-1)

    # vertical line
    vertical_tolerance = math.pi/180 * 0.1  # 0.1 degrees
    vertical_angle = math.pi/2

    if math.isclose(avg_angle, vertical_angle, abs_tol=vertical_tolerance):

        # Check approx. direction
        delta_y_first_to_last = points_sorted[-1][1].points_sorted[0][1]

        if delta_y_first_to_last >= 0:
            # top to bottom
            return math.inf, avg_x
        else:
            # bottom to top
            return -math.inf, avg_x

    # "Normal" line > return gradient and displacement
    gradient = math.atan(avg_angle)
    displacement = avg_y - (avg_x * gradient)

    return gradient, displacement


def line_angle(line):
    """
    Translates the gradient into an angle.

    In case of a vertical line, the angle will be either +/- 90° in radians (pi/2 resp. -pi/2).

    :param line: Line.
    :return: Angle of line.
    """
    if math.isclose(line[0], math.inf):
        return math.pi/2
    elif math.isclose(line[0], -math.inf):
        return -math.pi/2
    else:
        return math.atan(line[0])


def line_to_points(img, line):
    """
    Transforms line into two points so that the line crosses the whole image.

    Can handle vertical lines represented by tuple (inf, x displacement).

    :param img: Image to obtain dimensions from.
    :param line: Line to be represented by two points.
    :return: Tuple of two points.
    """
    dims = img.shape
    height = dims[0]
    width = dims[1]

    pt1 = None
    pt2 = None

    if math.isclose(line[0], math.inf):
        # vertical line, displacement = x
        pt1 = (int(round(line[1])), 0)
        pt2 = (int(round(line[1])), height-1)
    else:
        pt1 = (0, round(int(line[1])))
        pt2 = (width-1, int(round(line[0]*(width-1)+line[1])))

    return pt1, pt2


def calc_line(x, line):
    """
    Calculates y for given x of line.

    Can handle vertical lines represented by tuple (inf, x). In that case
    math.nan is returned.

    :param x: x value.
    :param line: Line tuple (gradient, y displacement).
    :return: y value or math.nan for vertical lines.
    """
    if math.isclose(line[0], math.inf):
        return math.nan
    else:
        return line[0] * x + line[1]


def draw_line(img, line, color=(255, 0, 0), thickness=1):
    """
    Draws a line that is represented by (theta, displacement).

    Can also draw vertical lines represented by (inf, x).

    :param img: Image to draw to.
    :param line: Line.
    :param color: Color.
    :param thickness: trivial..
    """

    (pt1, pt2) = line_to_points(img, line)
    cv2.line(img, pt1, pt2, color, thickness)