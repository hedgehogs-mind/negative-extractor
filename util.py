import statistics
import math

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


def n_closest_contours(self_contour, other_contours, n=1, output_contour_indices=True):
    """
    Find the closest contours. Therefore the contour centers are used. The contours are sorted by distance asc.

    'n' may be greater than the amount of contours. In this case, other_contours are returned,
    sorted by distance asc, but not more contours.

    :param self_contour: Contour to search closest contour for.
    :param other_contours: Contours to check.
    :param n: How many closest contours shall be retrieved.
    :param output_contour_indices: If true, the indices of the closest contours in the given list will be returned too.
    :return: Tuple (closest_contours, distances[, contour_indices]).
    """
    assert len(other_contours) > 0, "Need at least one other contour"

    self_center = contour_center(self_contour)

    # Max out each position
    distances = [math.inf] * n
    contour_indices = [-1] * n

    # now check for each contour if it has a distance lower than anything in the list
    for index, contour in enumerate(other_contours):
        center = contour_center(contour)
        dist = np.linalg.norm(self_center-center)

        # lower distance at any position: insert and shift, last ones will be popped
        for i in range(0, n):
            if dist < distances[i]:
                # Shift
                distances.insert(i, dist)
                contour_indices.insert(i, index)

                # Remove last ones
                distances.pop()
                contour_indices.pop()

                break

    # Check how many we actually found
    found_counter = 0
    for index in contour_indices:
        if index >= 0:
            found_counter += 1

    existing_contours_indices = contour_indices[:found_counter]
    existing_distances = distances[:found_counter]
    existing_closest_contours = list(map(lambda i: other_contours[i], existing_contours_indices))

    if output_contour_indices:
        return existing_closest_contours, existing_distances, existing_contours_indices
    else:
        return existing_closest_contours, existing_distances


def closest_transitive_contours(root_contour, other_contours, n=1):
    """
    Finds all contours that are close to the root (also in a transitive way)

    Example: B is close to A by distance 10. C is close to be B by 11. C is in
    the same group as B and A, Now comes D. It is close to C by 24. The average
    distance until now is 10.5, but the new distance 24 is greater than
    1.5 times the avg distance (here 15,75).

    At that point computation breaks and the group of "close" contours as well as the
    rest of the contours is returned.

    Uses method 'n_closest_contours()' internally. Due to this, you can vary the parameter 'n'.
    For more details, see doc of that method.

    :param root_contour: Root.
    :param other_contours: Other contours.
    :param n: Number of closest contours that shall be retrieved for a current fix point.
    :return: Tuple (close_contours, rest). Both lists of contours.
    """
    assert len(other_contours) > 0, "Need at least one other contour"

    rest = list(other_contours)
    group = list()

    distance_sum = 0
    contour_counter = 1  # our root
    fix_points = [root_contour]

    # Retrieve closests as long as we have fix points
    while len(fix_points) > 0 and len(rest) > 0:

        # all points we add become our new fix points
        new_fix_points = []

        for fix_point in fix_points:
            # find closests for given fix point
            contours, distances, indices = n_closest_contours(fix_point, rest, n)

            # we will delete them from the rest afterwards, so we just track what we added / need to delete
            added_contours_indices = []

            # Check every close contour if it shall be added
            for i in range(0, len(indices)):

                # calc avg distance based on current distances
                avg_distance = 0 if contour_counter < 2 else distance_sum / (contour_counter-1)
                distance = distances[i]

                contour_index = indices[i]
                contour = contours[i]

                # Add point of only one yet or distance is not larger than 1.5 times the avg. distance
                if contour_counter < 2 or distance < avg_distance * 1.5:

                    # add close contour
                    group.append(contour)
                    new_fix_points.append(contour)

                    # remember it to be deleted from list rest
                    added_contours_indices.append(contour_index)

                    contour_counter += 1
                    distance_sum += distance

            # remove from back to front to prevent unwanted shifts
            added_contours_indices.sort(reverse=True)

            for index_to_delete in added_contours_indices:
                rest.pop(index_to_delete)

            # If nothing left, break
            if len(rest) == 0:
                break

        # place new fix points
        fix_points = new_fix_points

    return group, rest


def group_contours_by_distance(contours, n=1):
    """
    This method groups the contours into individual groups based on average distance to each other.

    A group consists of contours that are close to each other by transitive relationship.
    Their distances are nearly the same.

    Uses method 'closest_transitive_contours()' internally. Due to this, you can also vary the
    parameter 'n' to find results, that suit you need better.
    For more details, check out doc of that method.

    :param contours: List of all contours.
    :param n: Max number of close contours to retrieve one at a time.
    :return: List of groups. Each group is a list of contours.
    """

    # We will sort them into groups
    groups = []
    holes = list(contours)

    while len(holes) > 0:
        root = holes.pop(0)
        close_contours, rest = closest_transitive_contours(root, holes, n)

        group = []
        group.append(root)
        group.extend(close_contours)

        groups.append(group)
        holes = rest

    return groups


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


def contours_center_line(contours):
    """
    Computes for each contour the center and uses the center points to create
    a line.

    :param contours: Contours to create line through their centers.
    :return: Line (gradient, y displacement).
    """
    centers = list(map(lambda cnt: contour_center(cnt), contours))
    return points_to_line(centers)


def contours_top_line(contours):
    """
    Creates line that runs along the top side of all contours.

    To make this work properly, the contours must be arranged in one line.

    :param contours: Contours.
    :return: Line (gradient, y displacement).
    """
    tops = list(map(lambda cnt: contour_top(cnt), contours))
    return points_to_line(tops)


def contours_bottom_line(contours):
    """
    Creates line that runs along the bottom side of all contours.

    To make this work properly, the contours must be arranged in one line.

    :param contours: Contours.
    :return: Line (gradient, y displacement).
    """
    bottoms = list(map(lambda cnt: contour_bottom(cnt), contours))
    return points_to_line(bottoms)


def most_left_contour(contours):
    """
    Searches the contour which has the most extreme left point.

    :param contours: Contours.
    :return: Most extreme contour.
    """

    min_x = math.inf
    left = None
    for contour in contours:
        cl = contour_left(contour)[0]

        if cl < min_x:
            left = contour
            min_x = cl

    return left


def most_right_contour(contours):
    """
    Searches the contour which has the most extreme right point.

    :param contours: Contours.
    :return: Most extreme contour.
    """

    max_x = -math.inf
    right = None
    for contour in contours:
        cr = contour_right(contour)[0]

        if cr > max_x:
            right = contour
            max_x = cr

    return right


def get_k_colors(img, k):
    """
    Returns k most dominant colors via k-means algorithm.

    :param img: Image to retrieve k colors from.
    :param k: Number of colors.
    :return: List of colors. Colors are numpy 16bit unsigned integers arrays.
    """
    data = img.reshape((-1, 3))
    data = np.float32(data)
    iterations = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, iterations, cv2.KMEANS_RANDOM_CENTERS)

    return list(map(lambda col: np.array([int(col[0]), int(col[1]), int(col[2])], dtype=np.uint16), center))


def sort_colors_by_brightness(colors):
    """
    Averages all colors and sort them by brightness ascending.

    :param colors: List of 3 value arrays/tuples.
    :return: Colors sorted by brightness ascending.
    """
    col_and_avgs = list(
        map(
            lambda col: (col, (col[0]+col[1]+col[2])),  # We just need it for sorting, so no /3 needed
            colors
        )
    )

    # Sort by average value (gray value)
    sorted_cols = sorted(col_and_avgs, key=lambda tup: tup[1])

    # Return only colors
    return list(map(lambda tup: tup[0], sorted_cols))


def calc_color_mask_diff(darkest_color, brightest_color):
    """
    TODO: doc pending peter .... :)

    :param darkest_color:
    :param brightest_color:
    :return:
    """
    darkest_color_avg = np.int16((darkest_color[0] + darkest_color[1] + darkest_color[2]) / 3)
    brightest_color_avg = np.int16((brightest_color[0] + brightest_color[1] + brightest_color[2]) / 3)

    dark_diff = np.array(darkest_color, dtype=np.int16) - darkest_color_avg
    brightest_diff = np.array(brightest_color, dtype=np.int16) - brightest_color_avg

    print(dark_diff)
    print(brightest_diff)

    # todo: Do I need both diffs?

    return brightest_diff