import math

import cv2
import time
import numpy as np

from util.contour import rectangle_corners, sort_contours_by_area
from util.image import add_border, blur, bw
from util.line import points_to_line, draw_line, line_angle

window = 'negative'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 800, 600)


#negative = cv2.imread("images/test_negative_small_rotated_mirrored.tiff")
#negative = cv2.imread("images/test_negative_small_rotated.tiff")
#negative = cv2.imread("images/test_single_negative_small.tiff")
#negative = cv2.imread("images/test_negative_small.tiff")
#negative = cv2.imread("images/test_negative.tiff")

dir = "images/ektar_16/"
#negative = cv2.imread(dir + "ektar_16bit_01_s.tif", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
negative = cv2.imread(dir + "ektar_16bit_02_s.tif", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)


# Processing start
t_start = time.time()


def extract_biggest_outer_contour(bordered_negative):
    """
    Applies slight blur and threshold to retrieve contours. The biggest will be returned.
    In case no one has been found, an assertion error will be raised.

    :param bordered_negative: Negative having a white border/background around strip.
    :return: Biggest contour.
    """

    # low absolute blur size reduces round corners in bw > corners will produce better lines
    blurred = blur(bordered_negative, 0, 3)
    bw_img = bw(blurred, mode=cv2.THRESH_BINARY_INV)

    # Let us find the contour of the strip
    cnts = sort_contours_by_area(
        cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],  # we dont need hierarchy
        direction=1
    )
    assert len(cnts) > 0, "Found no contours"
    return cnts[0]  # we assume, that the biggest contour is our strip


def shrink_corners_for_min_rectangle(rect_corners, additional_inset=0):
    """
    Creates a rectangle, that sits within the given 4 points.
    In case you want to further shrink the rectangle, you can pass an additional inset parameter.

    :param rect_corners: Four corners of rectangle in order (top left, top right, bottom right, bottom left).
    :param additional_inset: Amount by which the rectangle shall further shrink in each direction.
    :return: Numpy array 4x2 of type int. Point order matches the input order.
    """
    assert len(rect_corners) == 4, "Need 4 corners, got only {}".format(len(rect_corners))

    left_boundary = max(rect_corners[0][0], rect_corners[3][0])  # x of top left or bottom left?
    right_boundary = min(rect_corners[1][0], rect_corners[2][0])  # x of top right or bottom right?

    top_boundary = max(rect_corners[0][1], rect_corners[1][1])  # y of top left or right corner?
    bottom_boundary = min(rect_corners[3][1], rect_corners[2][1])  # y of bottom left or right corner?

    left_boundary += additional_inset
    right_boundary -= additional_inset

    top_boundary += additional_inset
    bottom_boundary -= additional_inset

    left_boundary = int(left_boundary)
    right_boundary = int(right_boundary)
    top_boundary = int(top_boundary)
    bottom_boundary = int(bottom_boundary)

    points = np.array([
        [left_boundary, top_boundary],  # top left corner
        [right_boundary, top_boundary],  # top right corner
        [right_boundary, bottom_boundary],  # bottom right corner
        [left_boundary, bottom_boundary]  # bottom left corner
    ]).reshape((4, 2))

    return points


def extract_strip(negative):
    # First step: extract plain strip
    bordered = add_border(negative, (255, 255, 255))

    strip_contour = extract_biggest_outer_contour(bordered)

    # Get the corners
    strip_corners = rectangle_corners(strip_contour)

    top_corners_ltr = strip_corners[:2]  # left to right
    bottom_corners_rtl = strip_corners[2:4]  # right to left

    top_line = points_to_line(top_corners_ltr)
    bottom_line = points_to_line(bottom_corners_rtl)

    avg_angle = line_angle((0.5 * (top_line[0] + bottom_line[0]), 0))
    avg_angle_degrees = math.degrees(avg_angle)

    rotation_tolerance_degrees = 0.5
    rotation_needed = math.fabs(avg_angle_degrees) > rotation_tolerance_degrees

    # If rotation is needed, let us rotate it
    if rotation_needed:

        # Now let us rotate the image
        #  > we do not need to resize the image, through rotation only
        #    strip "spikes" on the left and right vanishes behind the borders
        (h, w) = bordered.shape[:2]
        center = (cX, cY) = (w // 2, h // 2)
        m_rot = cv2.getRotationMatrix2D(center, avg_angle_degrees, 1.0)

        border_color = (np.iinfo(negative.dtype).max,) * 3

        # We can rotate the original image > border is everywhere the same,
        # computed angle works for original image too
        rotated_negative = cv2.warpAffine(
            bordered, m_rot, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_color
        )

        # Now we need to update some images
        bordered = add_border(rotated_negative, (255, 255, 255))
        strip_contour = extract_biggest_outer_contour(bordered)
        strip_corners = rectangle_corners(strip_contour)

    # todo > make inset proportional to size?
    min_rect_corner = shrink_corners_for_min_rectangle(strip_corners, additional_inset=3)
    top_boundary = min_rect_corner[0][1]
    bottom_boundary = min_rect_corner[3][1]

    left_boundary = min_rect_corner[0][0]
    right_boundary = min_rect_corner[1][0]

    return bordered[top_boundary:bottom_boundary, left_boundary:right_boundary]



strip = extract_strip(negative)



t_end = time.time()
print("time: {:.3f}s".format((t_end-t_start)))
# Processing end





cv2.imshow(window, strip)
cv2.waitKey(-1)
cv2.destroyAllWindows()

