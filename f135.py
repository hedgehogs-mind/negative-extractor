import math

import cv2

from strip import create_bordered_negative, create_bw_negative, get_sprocket_holes_contours, split_sprocket_holes
from util import contours_center_line, line_angle


def straighten_35mm_negative(negative):
    """
    Computes sprocket holes and uses them to calculate strip rotation.

    This method fixes the strip rotation and returns an image with white borders.
    This way you will always have a white background within the image.

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
    M_rot = cv2.getRotationMatrix2D(center, strip_angle_degrees, 1.0)

    # We can rotate the original image > border is everywhere the same,
    # computed angle works for original image too
    rotated_negative = cv2.warpAffine(
        bordered_negative, M_rot, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    rotated_bordered_negative = create_bordered_negative(rotated_negative)

    return rotated_bordered_negative
