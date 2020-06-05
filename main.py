from time import sleep

import cv2
import time
import math
import numpy as np

from f135 import straighten_35mm_negative, get_35mm_strip_colors, get_35mm_strip_top_border_coords, \
    get_35mm_strip_bottom_border_coords
from strip import create_bordered_negative, create_bw_negative, get_sprocket_holes_contours, split_sprocket_holes, \
    get_average_sprocket_hole_size
from util import draw_line, group_contours_by_distance, closest_transitive_contours, contours_top_line, \
    contours_bottom_line, contours_center_line, n_closest_contours, line_angle, most_left_contour, most_right_contour, \
    contour_center, points_to_line, get_k_colors, sort_colors_by_brightness, calc_color_mask_diff

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

# Compute colors for white balance
darkest_color, brightest_color = get_35mm_strip_colors(rotated_negative)

# Let us do the white balance :)
wb_negative = rotated_negative.copy()

color_mask = calc_color_mask_diff(darkest_color, brightest_color)
color_correction = -color_mask

print("Mask: {}".format(color_mask))
print("Diff: {}".format(color_correction))

# Apply color correction

wb_negative = wb_negative.astype(dtype=np.int16)
wb_negative[:, :] += color_correction
wb_negative = np.clip(wb_negative, 0, 255)
wb_negative = wb_negative.astype(dtype=np.uint8)


# And invert
wb_negative = cv2.bitwise_not(wb_negative)

# Now let us handle the contrast
pos_brightest_color = 255-(darkest_color+color_correction)
pos_darkest_color = 255-(brightest_color+color_correction)

color_displacement = np.mean(pos_darkest_color)
color_factor = 255/(np.mean(pos_brightest_color)-color_displacement)

print("Pos. brightest: {}".format(pos_brightest_color))
print("Pos. darkest:   {}".format(pos_darkest_color))

print("Displacement: {}".format(color_displacement))
print("Factor: {}".format(color_factor))


wb_negative = wb_negative.astype(dtype=np.int16)

wb_negative[:, :] = (wb_negative[:, :]-color_displacement) * color_factor

wb_negative = np.clip(wb_negative, 0, 255)
wb_negative = wb_negative.astype(dtype=np.uint8)

t_end = time.time()
print("time: {:.3f}s".format((t_end-t_start)))
# Processing end


cv2.imshow(window, wb_negative)
cv2.waitKey(0)

cv2.destroyAllWindows()

