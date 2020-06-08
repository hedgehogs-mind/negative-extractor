import cv2
import time

from util.image import add_border, blur, bw

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
blurred = blur(bordered)
bw_img = bw(blurred, mode=cv2.THRESH_BINARY_INV)



t_end = time.time()
print("time: {:.3f}s".format((t_end-t_start)))
# Processing end


(cnts, _) = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = cnts[0]

for i in range(0, 501):

    img = bordered.copy()

    epsilon = i/10
    poly = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(img, [poly], -1, (255, 0, 255), 6)
    print("epsilon: {}, #pts: {}".format(epsilon, len(poly)))

    cv2.imshow(window, img)
    cv2.waitKey(10)


cv2.waitKey(-1)
cv2.destroyAllWindows()

