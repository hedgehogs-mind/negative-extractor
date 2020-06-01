import statistics
import math

import cv2
import numpy as np


def points_to_line(points):
    """
    Creates line based on points. Averages lines.

    Points will be treated left to right.

    Descending gradients will result in positive theta (coordinate system is flipped y-wise).

    If the line is vertical, theta will be math.inf.

    :param points: List of points (x, y).
    :return: Tuple (gradient, y displacement) or (inf, x displacement) for vertical lines.
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

    gradient = 0
    displacement = 0

    # First check if we need to handle vertical lines
    if math.isclose(delta, math.inf):
        gradient = math.inf
        displacement = avg_x
    else:
        gradient = math.atan(delta)
        displacement = avg_y - (avg_x*delta)

    return gradient, displacement


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
    :return: numpy array [cx, cy].
    """
    M = cv2.moments(contour)
    cx = M['m10']/M['m00']
    cy = M['m01']/M['m00']

    return np.array([cx, cy])


def closest_contour(self_contour, other_contours, output_contour_index=True):
    """
    Find the closest contour. Therefore the contour centers are used.

    :param self_contour: Contour to search closest contour for.
    :param other_contours: Contours to check.
    :param output_contour_index: If true, the index of the closest contour in the given list is returned too.
    :return: Tuple (closest_contour, distance[, contour_index]).
    """
    assert len(other_contours) > 0, "Need at least one other contour"

    self_center = contour_center(self_contour)

    lowest_distance = math.inf
    nearest_contour_index = -1

    for index, contour in enumerate(other_contours):
        center = contour_center(contour)
        dist = np.linalg.norm(self_center-center)

        if dist < lowest_distance:
            lowest_distance = dist
            nearest_contour_index = index

    if output_contour_index:
        return other_contours[nearest_contour_index], lowest_distance, nearest_contour_index
    else:
        return other_contours[nearest_contour_index], lowest_distance


def closest_transitive_contours(root_contour, other_contours):
    """
    Finds all contours that are close to the root (also in a transitive way)

    Example: B is close to A by distance 10. C is close to be B by 11. C is in
    the same group as B and A, Now comes D. It is close to C by 24. The average
    distance until now is 10.5, but the new distance 24 is greater than
    1.5 times the avg distance (here 15,75).

    At that point computation breaks and the group of "close" contours as well as the
    rest of the contours is returned.

    :param root_contour: Root.
    :param other_contours: Other contours.
    :return: Tuple (close_contours, rest). Both lists of contours.
    """
    assert len(other_contours) > 0, "Need at least one other contour"

    rest = list(other_contours)
    group = list()

    group.append(root_contour)

    distance_sum = 0
    holes_in_group = 1  # our root
    last = root_contour

    for index, contour in enumerate(other_contours):
        closest, distance, closest_index = closest_contour(last, rest)

        curr_avg_distance = 0 if holes_in_group < 2 else distance_sum / (holes_in_group - 1)

        # Contour must be added, if we don not have a second one yet.
        #
        # Otherwise: The next contour belongs to the same group,
        # if the distance is not bigger than the double of the current average distance.

        if holes_in_group < 2 or distance < curr_avg_distance * 1.5:
            # Just add the contour
            group.append(contour)
            rest.pop(closest_index)

            distance_sum += distance
            holes_in_group += 1

            last = contour
        else:
            break

    return group, rest


def group_contours_by_distance(contours):
    """
    At least 4 contours are required.

    This method groups the contours into individual groups.

    A group consists of contours that are close to each other by transitive relationship.
    Their distances are nearly the same.

    Uses method 'closest_transitive_contours()' internally.

    :param contours: List of all contours.
    :return: List of groups. Each group is a list of contours.
    """

    assert len(contours) >= 4, "Need at least 4 contours, got only {}" \
        .format(len(contours))

    # We will sort them into groups
    groups = []
    group_counter = 0

    holes = list(contours)

    while len(holes):
        assert group_counter <= 2, "Got more than two groups / sprocket hole rows"

        root = holes.pop(0)

        group, rest = closest_transitive_contours(root, holes)

        groups.append(group)
        group_counter += 1

        holes = rest

    return groups


def contour_top(contour):
    return tuple(contour[contour[:, :, 1].argmin()][0])


def contour_bottom(contour):
    return tuple(contour[contour[:, :, 1].argmax()][0])


def contour_left(contour):
    return tuple(contour[contour[:, :, 0].argmin()][0])


def contour_right(contour):
    return tuple(contour[contour[:, :, 0].argmax()][0])
