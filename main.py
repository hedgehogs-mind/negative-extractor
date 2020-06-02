from time import sleep

import cv2
import time
import math
import numpy as np

from f135 import straighten_35mm_negative
from strip import create_bordered_negative, create_bw_negative, get_sprocket_holes_contours, split_sprocket_holes
from util import draw_line, group_contours_by_distance, closest_transitive_contours, contours_top_line, \
    contours_bottom_line, contours_center_line, n_closest_contours, line_angle

# todo: what happens if i have a negative with background all around?

window = 'negative'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 800, 600)





#negative = cv2.imread("images/test_negative_small_rotated_mirrored.tiff")
negative = cv2.imread("images/test_negative_small_rotated.tiff")
#negative = cv2.imread("images/test_single_negative_small.tiff")
#negative = cv2.imread("images/test_negative_small.tiff")
#negative = cv2.imread("images/test_negative.tiff")




# Processing start
t_start = time.time()

rotated_negative = straighten_35mm_negative(negative)

t_end = time.time()
print("time: {:.3f}s".format((t_end-t_start)))
# Processing end






cv2.imshow(window, negative)


cv2.waitKey(0)

cv2.imshow(window, rotated_negative)


cv2.waitKey(0)

cv2.destroyAllWindows()

