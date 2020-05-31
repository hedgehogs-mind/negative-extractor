from time import sleep

import cv2
import time

from strip import get_background_contours, get_background_contour_bottom_line, \
    get_background_contour_top_line, get_sprocket_holes_contours, get_background_border_lines, get_sprocket_holes_lines
from util import draw_line

# todo: what happens if i have a negative with background all around?

window = 'negative'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 800, 600)



t_start = time.time()


negative = cv2.imread("images/test_negative.tiff")
neg_copy = negative.copy()

grayscale_negative = cv2.cvtColor(negative, cv2.COLOR_RGB2GRAY)
blur = cv2.blur(grayscale_negative, (5, 5))
(thresh, bw_negative) = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY)


# Top and bottom line
(top_line, bottom_line) = get_background_border_lines(bw_negative)
draw_line(neg_copy, top_line, (255, 0, 255))
draw_line(neg_copy, bottom_line, (255, 0, 255))


# Let us retrieve the sprocket holes
(top_holes, bottom_holes) = get_sprocket_holes_contours(bw_negative, top_line, bottom_line)
cv2.drawContours(neg_copy, top_holes, -1, (0, 180, 0), 2)
cv2.drawContours(neg_copy, bottom_holes, -1, (180, 0, 0), 2)


# Let us retrieve the sprocket hole lines
(top_upper_line, top_lower_line) = get_sprocket_holes_lines(top_holes)
(bottom_upper_line, bottom_lower_line) = get_sprocket_holes_lines(bottom_holes)

draw_line(neg_copy, top_upper_line, (0, 0, 255))
draw_line(neg_copy, top_lower_line, (0, 0, 255))

draw_line(neg_copy, bottom_upper_line, (0, 0, 255))
draw_line(neg_copy, bottom_lower_line, (0, 0, 255))



t_end = time.time()
print("time: {}".format((t_end-t_start)))





cv2.imshow(window, neg_copy)


cv2.waitKey(0)
cv2.destroyAllWindows()

