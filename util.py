import statistics
import math

import cv2


def points_to_line(points):
    """
    Creates line based on points. Averages lines.

    Points will be treated left to right.

    Descending gradients will result in positive theta (coordinate system is flipped y-wise).

    If the line is vertical, theta will be math.inf.

    :param points: List of tuples (x, y).
    :return: Tuple (theta in radians, y displacement) or (inf, x displacement) for vertical lines.
    """

    assert len(points) > 1, "Need at least two points"

    points_ltr = sorted(points, key=lambda p: p[0])

    delta_sum = 0
    for i in range(2, len(points_ltr)-1):
        y1 = points_ltr[i-1][1]
        y2 = points_ltr[i][1]

        delta_sum += y2-y1

    dist_left_to_right = points_ltr[-1][0]-points_ltr[0][0]
    delta = delta_sum/dist_left_to_right
    avg_x = statistics.mean(map(lambda p: p[0], points))
    avg_y = statistics.mean(map(lambda p: p[1], points))

    theta = 0
    displacement = 0

    # First check if we need to handle vertical lines
    if math.isclose(delta, math.inf):
        theta = math.inf
        displacement = avg_x
    else:
        theta = math.atan(delta)
        displacement = avg_y - (avg_x*delta)

    return theta, displacement


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
        delta = math.tan(line[0])
        pt1 = (0, round(int(line[1])))
        pt2 = (width-1, int(round(delta*(width-1)+line[1])))

    return pt1, pt2


def calc_line(x, line):
    """
    Calculates y for given x of line.

    Can handle vertical lines represented by tuple (inf, x). In that case
    math.nan is returned.

    :param x: x value.
    :param line: Line tuple (theta in radians, y displacement).
    :return: y value or math.nan for vertical lines.
    """
    if math.isclose(line[0], math.inf):
        return math.nan
    else:
        gradient = math.tan(line[0])
        return gradient * x + line[1]


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


def draw_rect(img, rect, color=(255, 0, 0), thickness=1):
    """
    Draws a rectangle represented by (x, y, w, h).

    :param img: Image to draw to.
    :param rect:  Rectangle as tuple.
    :param color: trivial.
    :param thickness: trivial.
    """
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color, thickness)


def contour_center(contour):
    """
    Calculates center vie moments.

    :param contour: Contour to calc center for.
    :return: Tuple (cx, cy).
    """
    M = cv2.moments(contour)
    cx = M['m10']/M['m00']
    cy = M['m01']/M['m00']

    return cx, cy
