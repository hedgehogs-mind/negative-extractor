from time import sleep

import cv2
import time

from strip import get_background_contours, get_background_contour_bottom_line, \
    get_background_contour_top_line
from util import draw_line



window = 'negative'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 800, 600)



t_start = time.time()


negative = cv2.imread("images/test_negative_small_rotated.tiff")
neg_copy = negative.copy()

grayscale_negative = cv2.cvtColor(negative, cv2.COLOR_RGB2GRAY)
blur = cv2.blur(grayscale_negative, (5, 5))
(thresh, bw_negative) = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY)


(top, bottom) = get_background_contours(bw_negative)

if top is not None:
    cv2.drawContours(neg_copy, [top], -1, (0, 0, 255), 1)

    top_line = get_background_contour_bottom_line(bw_negative, top)
    draw_line(neg_copy, top_line, (0, 255, 0))

if bottom is not None:
    cv2.drawContours(neg_copy, [bottom], -1, (0, 0, 255), 2)
    bottom_line = get_background_contour_top_line(bw_negative, bottom)
    draw_line(neg_copy, bottom_line, (0, 255, 0))






t_end = time.time()
print("time: {}".format((t_end-t_start)))





cv2.imshow(window, neg_copy)


cv2.waitKey(0)
cv2.destroyAllWindows()

