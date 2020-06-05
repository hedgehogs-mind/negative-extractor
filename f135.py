import math

import cv2

from strip import create_bordered_negative, create_bw_negative, get_sprocket_holes_contours, split_sprocket_holes, \
    get_average_sprocket_hole_size
from util import contours_center_line, line_angle, contours_top_line, most_right_contour, most_left_contour, \
    contour_center, contours_bottom_line, get_k_colors, sort_colors_by_brightness

import numpy as np

border_start_dist_rel_to_hole_size = 0.12
border_end_dist_rel_to_hole_size = 0.45


def straighten_35mm_negative(negative):
    """
    Computes sprocket holes and uses them to calculate strip rotation.

    This method fixes the strip rotation and returns an image with white borders.
    This way you will always have a white background within the image.

    Supports any color depth.

    :param negative: Original negative image.
    :return: Image of straight negative with white background around it.
    """

    # First let us add a border
    #  > by that we always have an image with white background
    bordered_negative = create_bordered_negative(negative)

    # Makes background black and strip white
    bw_negative = create_bw_negative(bordered_negative)

    # Now let us find all sprocket hole contours within the strip
    sprocket_holes_contours = get_sprocket_holes_contours(bw_negative)

    # Let us divide the holes into top and bottom
    (top_holes, bottom_holes) = split_sprocket_holes(sprocket_holes_contours)

    # Let us find top and bottom line via sprocket holes and compute rot. angle
    tcl = contours_center_line(top_holes)
    bcl = contours_center_line(bottom_holes)

    angle_top = line_angle(tcl)
    angle_bottom = line_angle(bcl)
    strip_angle = 0.5 * (angle_top + angle_bottom)
    strip_angle_degrees = math.degrees(strip_angle)

    # Now let us rotate the image
    #  > we do not need to resize the image, through rotation only
    #    strip "spikes" on the left and right vanishes behind the borders
    (h, w) = bordered_negative.shape[:2]
    center = (cX, cY) = (w // 2, h // 2)
    m_rot = cv2.getRotationMatrix2D(center, strip_angle_degrees, 1.0)

    border_color = (np.iinfo(negative.dtype).max,) * 3

    # We can rotate the original image > border is everywhere the same,
    # computed angle works for original image too
    rotated_negative = cv2.warpAffine(
        bordered_negative, m_rot, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color
    )

    rotated_bordered_negative = create_bordered_negative(rotated_negative)

    return rotated_bordered_negative


def get_35mm_strip_top_border_coords(top_sprocket_holes):
    """
    Only works properly if the sprocket holes are horizontally aligned.

    Computes the rectangle between the sprocket holes at the top and
    the top border. Rect starts at the most left sprocket hole
    and ends at the hole on the right.

    The rectangle height is determined relative to the average sprocket
    hole height.

    An example:

    ========================================================== (border)

    (pt1)--------------------------------------------------o
     |             all of this is                          |
     |                     in the rectangle                |
     o-------------------------------------------------- (pt2)

    ####     ####     ####     ####     ####     ####     ####
    #  #     #  #     #  #     #  #     #  #     #  #     #  #
    #  #     #  #     #  #     #  #     #  #     #  #     #  #
    ####     ####     ####     ####     ####     ####     ####

    >> Here is the negative are


    :param top_sprocket_holes: Sprocket hole contours, sorted left to right.
    :return: Tuple (pt1, pt2). pt1 corner top left, pt2 bottom right. Points are tuples of (x, y).
    """

    top_line = contours_top_line(top_sprocket_holes)

    (top_hole_w, top_hole_h) = get_average_sprocket_hole_size(top_sprocket_holes)
    top_left_hole = most_left_contour(top_sprocket_holes)
    top_right_hole = most_right_contour(top_sprocket_holes)

    left_bound = int(math.ceil(contour_center(top_left_hole)[0]))
    right_bound = int(math.floor(contour_center(top_right_hole)[0]))

    bottom_bound = int(math.ceil(top_line[1] - top_hole_h * border_start_dist_rel_to_hole_size))
    top_bound = int(math.floor(bottom_bound - top_hole_h * border_end_dist_rel_to_hole_size))

    return (left_bound, top_bound), (right_bound, bottom_bound)


def get_35mm_strip_bottom_border_coords(bottom_sprocket_holes):
    """
    Only works properly if the sprocket holes are horizontally aligned.

    Computes the rectangle between the sprocket holes at the bottom and
    the bottom border. Rect starts at the most left sprocket hole
    and ends at the hole on the right.

    The rectangle height is determined relative to the average sprocket
    hole height.

    An example:

    >> Here is the negative are

    ####     ####     ####     ####     ####     ####     ####
    #  #     #  #     #  #     #  #     #  #     #  #     #  #
    #  #     #  #     #  #     #  #     #  #     #  #     #  #
    ####     ####     ####     ####     ####     ####     ####

     (pt1)--------------------------------------------------o
      |             all of this is                          |
      |                     in the rectangle                |
      o-------------------------------------------------- (pt2)

    =========================================================== (border)

    :param bottom_sprocket_holes: Sprocket hole contours, sorted left to right.
    :return: Tuple (pt1, pt2). pt1 corner top left, pt2 bottom right. Points are tuples of (x, y).
    """

    bottom_line = contours_bottom_line(bottom_sprocket_holes)

    (bottom_hole_w, bottom_hole_h) = get_average_sprocket_hole_size(bottom_sprocket_holes)
    bottom_left_hole = most_left_contour(bottom_sprocket_holes)
    bottom_right_hole = most_right_contour(bottom_sprocket_holes)

    left_bound = int(math.ceil(contour_center(bottom_left_hole)[0]))
    right_bound = int(math.floor(contour_center(bottom_right_hole)[0]))

    top_bound = int(math.ceil(bottom_line[1] + bottom_hole_h * border_start_dist_rel_to_hole_size))
    bottom_bound = int(math.floor(top_bound + bottom_hole_h * border_end_dist_rel_to_hole_size))

    return (left_bound, top_bound), (right_bound, bottom_bound)


def get_35mm_strip_colors(negative, positive=False):
    """
    Negative must have a white border/background all around!

    Sprocket holes need to be visible!

    Takes border area on top and bottom between sprocket holes and edge.
    Computes two most dominant colors within this area.
    Returns brightest and darkest color found in there.

    :param negative: Negative with white border.
    :param positive: If True, negative image will be inverted before processing
    :return: Returns tuple (darkest_color, brightest_color).
    """
    neg_copy = negative
    if positive:
        neg_copy = cv2.bitwise_not(neg_copy)

    bw = create_bw_negative(neg_copy)
    sprocket_holes = get_sprocket_holes_contours(bw)
    (top_holes, bottom_holes) = split_sprocket_holes(sprocket_holes)

    # Compute the border rectangles
    top_border_rect = get_35mm_strip_top_border_coords(top_holes)
    bottom_border_rect = get_35mm_strip_bottom_border_coords(bottom_holes)

    # Extract sub images
    roi_top = negative[
        top_border_rect[0][1]:top_border_rect[1][1],  # y1 and y2,
        top_border_rect[0][0]:top_border_rect[1][0]   # x1 and x2
    ]

    roi_bottom = negative[
        bottom_border_rect[0][1]:bottom_border_rect[1][1],  # y1 and y2,
        bottom_border_rect[0][0]:bottom_border_rect[1][0]   # x1 and x2
    ]

    # Now compute brightest and darkest color
    colors = list()
    colors.extend(get_k_colors(roi_top, 2))
    colors.extend(get_k_colors(roi_bottom, 2))

    # Sort colors
    sorted_colors = sort_colors_by_brightness(colors)

    return sorted_colors[0], sorted_colors[-1]