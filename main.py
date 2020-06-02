from time import sleep

import cv2
import time
import math
import numpy as np

from strip import create_bordered_negative, create_bw_negative, get_sprocket_holes_contours, split_sprocket_holes
from util import draw_line, group_contours_by_distance, closest_transitive_contours, contours_top_line, \
    contours_bottom_line, contours_center_line, n_closest_contours

# todo: what happens if i have a negative with background all around?

window = 'negative'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 800, 600)





#negative = cv2.imread("images/test_negative_small_rotated_mirrored.tiff")
negative = cv2.imread("images/test_negative_small_rotated.tiff")
#negative = cv2.imread("images/test_single_negative_small.tiff")
#negative = cv2.imread("images/test_negative_small.tiff")



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

cv2.drawContours(neg_copy, top_holes, -1, (255, 0, 0), 1)
cv2.drawContours(neg_copy, bottom_holes, -1, (0, 0, 255), 1)

# Let us draw their top, center and bottom lines

print("LOL:")
tul = contours_top_line(top_holes)
print("tul: {}".format(tul))
tcl = contours_center_line(top_holes)
tll = contours_bottom_line(top_holes)

bul = contours_top_line(bottom_holes)
bcl = contours_center_line(bottom_holes)
bll = contours_bottom_line(bottom_holes)

draw_line(neg_copy, tul, (230, 255, 0), 1)
draw_line(neg_copy, tcl, (230, 255, 0), 1)
draw_line(neg_copy, tll, (230, 255, 0), 1)

draw_line(neg_copy, bul, (230, 255, 0), 1)
draw_line(neg_copy, bcl, (230, 255, 0), 1)
draw_line(neg_copy, bll, (230, 255, 0), 1)




# Processing end

t_end = time.time()
print("time: {:.3f}s".format((t_end-t_start)))



# grayscale_negative = cv2.cvtColor(negative, cv2.COLOR_RGB2GRAY)
# blur = cv2.blur(grayscale_negative, (5, 5))
# (thresh, bw_negative) = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY)




# bw_inverted = cv2.bitwise_not(bw_negative)
#
# contours, hierarchy = cv2.findContours(bw_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(neg_copy, contours, -1, (255, 0, 0), 1)

# empty_image = bw_inverted * 0
# cv2.drawContours(empty_image, contours, 0, (255, 255, 255), 1)


# rect = cv2.minAreaRect(contours[0])
# rect_points = np.int0(cv2.boxPoints(rect))
#
# print(rect_points)
#
# cv2.drawContours(neg_copy, [rect_points], -1, (0, 0, 255), 2)

cv2.imshow(window, neg_copy)


cv2.waitKey(0)
cv2.destroyAllWindows()

