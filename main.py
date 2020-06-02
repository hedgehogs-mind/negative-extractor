from time import sleep

import cv2
import time
import math
import numpy as np

from strip import create_bordered_negative, create_bw_negative, get_sprocket_holes_contours, split_sprocket_holes
from util import draw_line, group_contours_by_distance, closest_transitive_contours, contours_top_line, \
    contours_bottom_line, contours_center_line, n_closest_contours, line_angle

# todo: what happens if i have a negative with background all around?

window = 'negative'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 800, 600)





#negative = cv2.imread("images/test_negative_small_rotated_mirrored.tiff")
#negative = cv2.imread("images/test_negative_small_rotated.tiff")
#negative = cv2.imread("images/test_single_negative_small.tiff")
#negative = cv2.imread("images/test_negative_small.tiff")
negative = cv2.imread("images/test_negative.tiff")




# Processing start
t_start = time.time()


# First let us add a white border
bordered_negative = create_bordered_negative(negative)
neg_copy = bordered_negative

bw_negative = create_bw_negative(bordered_negative)



# Now lets create contours with hierarchy
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

print("Strip angle: {}°".format(strip_angle_degrees))

(h, w) = neg_copy.shape[:2]
center = (cX, cY) = (w // 2, h // 2)
M_rot = cv2.getRotationMatrix2D(center, strip_angle_degrees, 1.0)
rot_neg_copy = cv2.warpAffine(neg_copy, M_rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))




# Processing end

t_end = time.time()
print("time: {:.3f}s".format((t_end-t_start)))



cv2.imshow(window, rot_neg_copy)


cv2.waitKey(0)
cv2.destroyAllWindows()

