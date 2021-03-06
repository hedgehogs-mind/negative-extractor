import math

import cv2
from util import group_contours_by_distance, points_to_line, contour_center
import numpy as np

border_size_rel_to_dims = 0.01
border_min_size = 4

blur_size_rel_to_dims = 0.001
blur_min_size = 2

bw_threshold_percent = 0.9


def create_bordered_negative(negative):
    """
    Adds 1% wide border around the negative. Border color is white.

    Supports any color depths.

    :param negative: Negative image.
    :return: Image with additional white border.
    """
    (h, w) = negative.shape[:2]
    pad = int(math.ceil(max(border_size_rel_to_dims * max(h, w), border_min_size)))

    border_color = (np.iinfo(negative.dtype).max,) * 3

    border_negative = cv2.copyMakeBorder(
        negative,
        pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=border_color
    )

    return border_negative


def create_bw_negative(negative):
    """
    The image will be made black and white. The negative (strip) will be white and
    the background black.

    Min blur size is 2, or 0.1% of max dims.

    Supports any color depth.

    :param negative: Negative image.
    :return: Negative as bw image. Background and holes are black, strip white.
    """

    (h, w) = negative.shape[:2]
    blur_size = int(math.ceil(max(blur_size_rel_to_dims * max(h, w), blur_min_size)))

    max_val = np.iinfo(negative.dtype).max
    threshold_val = max_val * bw_threshold_percent

    gray_negative = cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray_negative, (blur_size, blur_size))
    (thresh, bw_negative) = cv2.threshold(gray_blur, threshold_val, max_val, cv2.THRESH_BINARY_INV)

    return bw_negative


def get_sprocket_holes_contours(bw_negative):
    """
    Assumes the given image has a black background around the negative.
    The strip itself must be white.

    All contours within the negative will be returned (should only be the sprocket holes)

    Searches the biggest contour in the first hierarchy level (prevents dust) and then grabs
    all its children as sprocket holes.

    :param bw_negative: Negative with black background/border and white strip.
    :return: All contours found within the strip. Not arranged.
    """

    # findContours can only handle 8bit images
    if bw_negative.dtype is not np.uint8:
        bw8 = bw_negative.astype(np.uint8)

    contours, hierarchy = cv2.findContours(bw8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Grab size of first root contour, assume it is the biggest
    big_root = 0
    big_rect = cv2.minAreaRect(contours[big_root])
    big_size = big_rect[1][0] * big_rect[1][1]

    # check if there are contours in the same hierarchy level bigger than the first root
    next_root = hierarchy[0][big_root][0]
    while next_root >= 0:
        next_rect = cv2.minAreaRect(contours[next_root])
        next_size = next_rect[1][0] * next_rect[1][1]

        if next_size > big_size:
            big_root = next_root
            big_rect = next_rect
            big_size = next_size

        next_root = hierarchy[0][next_root][0]

    # First is the child
    child_contour = hierarchy[0][big_root][2]

    child_contours = []
    while child_contour >= 0:
        child_contours.append(contours[child_contour])
        child_contour = hierarchy[0][child_contour][0]

    return child_contours


def split_sprocket_holes(sprocket_holes_contours):
    """
    Splits the given sprocket holes into two groups: top and bottom holes.

    :param sprocket_holes_contours: All contours.
    :return: Tuple (top_contours, bottom_contours). Contours are sorted left to right.
    """

    # n = 2, because a sprocket hole can be at the left and/or right of the current hole
    groups = group_contours_by_distance(sprocket_holes_contours, 2)
    assert len(groups) == 2, "Expected two rows of sprocket holes, but found {}".format(len(groups))

    g1 = groups[0]
    g2 = groups[1]

    # First let us add the centers to the groups
    g1 = list(map(lambda cnt: (cnt, contour_center(cnt)), g1))
    g2 = list(map(lambda cnt: (cnt, contour_center(cnt)), g2))

    # Now let us sort the groups left to right
    #  > tup[1] is center ... [0] x coordinate
    g1 = sorted(g1, key=lambda tup: tup[1][0])
    g2 = sorted(g2, key=lambda tup: tup[1][0])

    # Get the list of centers
    centers1 = list(map(lambda tup: tup[1], g1))
    centers2 = list(map(lambda tup: tup[1], g2))

    line1 = points_to_line(centers1)
    line2 = points_to_line(centers2)

    top = None
    bottom = None

    # Line 1 is above line 2, [1] is y displacement
    if line1[1] < line2[1]:
        top = g1
        bottom = g2
    else:
        top = g2
        bottom = g1

    # Remove the centers
    top = list(map(lambda tup: tup[0], top))
    bottom = list(map(lambda tup: tup[0], bottom))

    return top, bottom


def get_average_sprocket_hole_size(sprocket_holes_contours):
    """
    Wraps rectangles around given contours and uses OpenCV's minAreaRect to
    get all widths and height. They will then be averaged.

    :param sprocket_holes_contours: Contours to get average width and height from.
    :return: Tuple (avg_width, avg_height). Dimensions are returned as floats.
    """

    width_sum = 0
    height_sum = 0

    for hole in sprocket_holes_contours:
        rect = cv2.minAreaRect(hole)
        width_sum += rect[1][0]
        height_sum += rect[1][1]

    avg_width = width_sum / len(sprocket_holes_contours)
    avg_height = height_sum / len(sprocket_holes_contours)

    return avg_width, avg_height
