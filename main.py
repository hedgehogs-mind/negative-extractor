from time import sleep

import cv2
import time
import math
import numpy as np

from f135 import straighten_35mm_negative
from strip import create_bordered_negative, create_bw_negative, get_sprocket_holes_contours, split_sprocket_holes, \
    get_average_sprocket_hole_size
from util import draw_line, group_contours_by_distance, closest_transitive_contours, contours_top_line, \
    contours_bottom_line, contours_center_line, n_closest_contours, line_angle, most_left_contour, most_right_contour, \
    contour_center, points_to_line, get_k_colors, sort_colors_by_brightness

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

rotated_negative = straighten_35mm_negative(negative)


# todo: move all this into "f135" > get strip colors

# Now again, get sprocket holes
bw = create_bw_negative(rotated_negative)
sprocket_holes = get_sprocket_holes_contours(bw)
(top_holes, bottom_holes) = split_sprocket_holes(sprocket_holes)

# Treat top
top_line = contours_top_line(top_holes)

top_hole_size = (top_hole_w, top_hole_h) = get_average_sprocket_hole_size(top_holes)
top_left_hole = most_left_contour(top_holes)
top_right_hole = most_right_contour(top_holes)

# Needed to evaluate top and bottom bound relative to the holes
dist_rel_to_hole_size = 0.12
grab_height_rel_to_hole_size = 0.45

left_bound = int(math.ceil(contour_center(top_left_hole)[0]))
right_bound = int(math.floor(contour_center(top_right_hole)[0]))

bottom_bound = int(math.ceil(top_line[1] - top_hole_h * dist_rel_to_hole_size))
top_bound = int(math.floor(bottom_bound - top_hole_h * grab_height_rel_to_hole_size))

outside_top = rotated_negative[top_bound:bottom_bound, left_bound:right_bound]

# Now compute colors
colors = get_k_colors(outside_top, 2)
sorted_colors = sort_colors_by_brightness(colors)

darkest_color = sorted_colors[0]
brightest_color = sorted_colors[-1]

cv2.circle(rotated_negative, (200, 90), 30, darkest_color, -1)
cv2.circle(rotated_negative, (200, 90), 30, (255, 0, 0), 1)

cv2.circle(rotated_negative, (400, 90), 30, brightest_color, -1)
cv2.circle(rotated_negative, (400, 90), 30, (255, 0, 0), 1)

# Draw colors



t_end = time.time()
print("time: {:.3f}s".format((t_end-t_start)))
# Processing end






cv2.imshow(window, rotated_negative)
cv2.waitKey(0)

cv2.destroyAllWindows()

