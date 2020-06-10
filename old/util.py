import statistics
import math

import cv2
import numpy as np





def draw_rect(img, rect, color=(255, 0, 0), thickness=1):
    """
    Draws a rectangle represented by (x, y, w, h).

    :param img: Image to draw to.
    :param rect:  Rectangle as tuple.
    :param color: trivial.
    :param thickness: trivial.
    """
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color, thickness)

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

    Handles any color depth.

    :param img: Image to retrieve k colors from.
    :param k: Number of colors.
    :return: List of colors. Colors are numpy arrays..
    """
    data = img.reshape((-1, 3))

    data = np.float32(data)
    iterations = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, iterations, cv2.KMEANS_RANDOM_CENTERS)

    return list(map(lambda col: np.array([col[0], col[1], col[2]], dtype=img.dtype), center))


def sort_colors_by_brightness(colors):
    """
    Averages all colors and sort them by brightness ascending.

    :param colors: List of 3 value arrays/tuples.
    :return: Colors sorted by brightness ascending.
    """
    def color_weight(col):
        f_col = np.float32(col)
        return f_col[0] + f_col[1] + f_col[2]

    col_and_avgs = list(
        map(
            lambda col: (col, color_weight(col)),  # We just need it for sorting, so no /3 needed
            colors
        )
    )

    # Sort by average value (gray value)
    sorted_cols = sorted(col_and_avgs, key=lambda tup: tup[1])

    # Return only colors
    return list(map(lambda tup: tup[0], sorted_cols))


def calc_white_balance_diff(color):
    """
    Handles any color depth.

    :param color: Numpy uint(8/16/..) array.
    :return: Color difference always as signed integer 64bit numpy array!
    """

    f_color = np.float32(color)
    f_avg = (f_color[0] + f_color[1] + f_color[2]) / 3

    f_diff = f_color - f_avg
    f_diff = np.round(f_diff)

    return np.array(f_diff, dtype=np.int64)