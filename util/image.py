import math

import cv2
import numpy as np


def add_border(image, color, percentage=0.01, minimum=1):
    """
    Adds a border to the given image. Supports 8bit and 16bit images.

    The border with is relative to the width or height of the image,
    the greatest of those two is selected and multiplied by the given
    percentage. If the resulting width is lower than the given minimum
    value, the latter is used instead.

    :param image: Image to add border to.
    :param color: Tuple of border color (BGR). Values only in range 0-255.
    :param percentage: Border width relative to the biggest image length.
    :return: Original image with border around it. Image will be bigger than the original.
    """
    assert minimum >= 1, "Min border width must be > 0"
    assert percentage >= 0, "Only positive percentage is allowed"

    border_color = np.asarray(color)

    if image.dtype == np.uint16:
        border_color = border_color.astype(np.uint32)
        shifted = border_color + 1
        stretched = shifted * 256
        border_color = stretched - 1  # normalized again

    border_color = (
        int(border_color[0]),
        int(border_color[1]),
        int(border_color[2])
    )

    (h, w) = image.shape[:2]
    max_dim = max(h, w)
    border_with = int(max(minimum, math.ceil(max_dim * percentage)))

    return cv2.copyMakeBorder(
        image,
        border_with, border_with, border_with, border_with,
        cv2.BORDER_CONSTANT,
        value=border_color
    )


def blur(image, mode=1, value=0.005, minimum=1):
    """
    Explicit blur size.

    OR

    Blurs the given image relative to its size. The value of width or height is used
    (the greater one) and multiplied by the given percentage. Is the resulting value than
    the minimum value.

    :param image: Image to blur.
    :param mode: 0 for absolute blur size (value param) or 1 for relative blur size.
    :param value: Absolute blur size or blur size relative to the biggest image dimension.
    :param minimum: Minimum blur size (>= 1).
    :return: Blurred image.
    """

    assert mode == 0 or mode == 1, "Mode must be either 0 or 1"
    assert value >= 0, "Value must be >= 0"
    assert minimum >= 1, "Min must be at least 1"

    blur_size = minimum
    if mode == 0:
        blur_size = max(blur_size, int(value))
    else:
        (h, w) = image.shape[:2]
        max_dim = max(h, w)
        blur_size = int(max(minimum, math.ceil(max_dim * value)))

    return cv2.blur(image, (blur_size, blur_size))


def bw(image, threshold=0.9, mode=cv2.THRESH_BINARY):
    """
    Turns an image into a black and white image (bi color/binary).

    :param image: Must be BGR format. Image to be turned bi color black/white.
    :param threshold: Threshold value relative to image max value (makes it bit depth independent).
    :param mode: Threshold mode used for cv2.threshold function.
    :return: 8bit Black and white image. White parts stay white, everything lower than white*percentage is black.
    """
    max_value = np.iinfo(image.dtype).max
    threshold_value = int(threshold * max_value)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (_, bw_img) = cv2.threshold(gray, threshold_value, max_value, mode)

    return bw_img.astype(np.uint8)
