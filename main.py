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

# First step: extract plain strip
bordered = add_border(negative, (255, 255, 255))

# low absolute blur size reduces round corners in bw > corners will produce better lines
blurred = blur(bordered, 0, 3)
bw_img = bw(blurred, mode=cv2.THRESH_BINARY_INV)

(cnts, _) = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sort_contours_by_area(cnts, direction=1)
assert len(cnts) > 0, "Found no contours"
strip_contour = cnts[0]  # we assume, that the biggest contour is our strip

# Now let us extract the corners
corners = rectangle_corners(strip_contour)
print("Corners: {}".format(corners))

ltr_corners = sorted(corners, key=lambda pt: pt[0])
left_corners = sorted(ltr_corners[:2], key=lambda pt: pt[1])
right_corners = sorted(ltr_corners[2:], key=lambda pt: pt[1])

top_corners = [left_corners[0], right_corners[0]]
bottom_corners = [left_corners[1], right_corners[1]]

top_line = points_to_line(top_corners)
bottom_line = points_to_line(bottom_corners)

avg_angle = line_angle((0.5 * (top_line[0] + bottom_line[0]), 0))
avg_angle_degrees = math.degrees(avg_angle)
print("Angle rad.: {}".format(avg_angle))
print("Angle deg.: {}".format(math.degrees(avg_angle)))

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



#cv2.drawContours(bordered, [left_corners], -1, (0, 0, 65535), 10)


t_end = time.time()
print("time: {:.3f}s".format((t_end-t_start)))
# Processing end






cv2.imshow(window, rotated_negative)
cv2.waitKey(-1)
cv2.destroyAllWindows()

