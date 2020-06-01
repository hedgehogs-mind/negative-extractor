from statistics import mean

import cv2

from util import closest_contour, group_contours_by_distance, points_to_line, contour_center


def create_bordered_negative(negative):
    """
    Adds two pixel wide border around the negative.

    Border color is white.

    :param negative: Negative image.
    :return: Image with additional white border.
    """
    border_padding = 2

    border_negative = cv2.copyMakeBorder(
        negative,
        border_padding, border_padding, border_padding, border_padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    return border_negative


def create_bw_negative(negative):
    """
    The image will be made black and white. The negative (strip) will be white and
    the background black.

    :param negative: Negative image.
    :return: Negative as bw image. Background and holes are black, strip white.
    """

    gray_negative = cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray_negative, (2, 2))
    (thresh, bw_negative) = cv2.threshold(gray_blur, 230, 255, cv2.THRESH_BINARY_INV)

    return bw_negative


def get_sprocket_holes_contours(bw_negative):
    """
    Assumes the given image has a black background around the negative.
    The strip itself must be white.

    All contours within the negative will be returned (should only be the sprocket holes)

    :param bw_negative: Negative with black background/border and white strip.
    :return: All contours found within the strip.
    """

    contours, hierarchy = cv2.findContours(bw_negative, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    next_of_root = hierarchy[0][0][0]
    assert next_of_root == -1, "There are more than one rectangles in the first hierarchy level"

    # First is the child
    next_contour = hierarchy[0][0][2]

    child_contours = []
    while next_contour >= 0:
        child_contours.append(contours[next_contour])
        next_contour = hierarchy[0][next_contour][0]

    return child_contours


def split_sprocket_holes(sprocket_holes_contours):
    """
    Splits the given sprocket holes into two groups: top and bottom holes.

    :param sprocket_holes_contours: All contours.
    :return: Tuple (top_contours, bottom_contours).
    """

    groups = group_contours_by_distance(sprocket_holes_contours)
    assert len(groups) == 2, "Expected two rows of sprocket holes, but found {}".format(len(groups))

    g1 = groups[0]
    g2 = groups[1]

    centers1 = list(map(lambda cnt: contour_center(cnt), g1))
    centers2 = list(map(lambda cnt: contour_center(cnt), g2))

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

    return top, bottom


