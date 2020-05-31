import cv2

from util import points_to_line


def get_background_contours(bw):
    """
    Tries to retrieve the areas above and below the negative strip (the background).
    If not found, None will be returned.

    :param bw: Strip as b/w image. Strip must be dark, sprocket holes and "background" white.
    :return: Tuple (top_contour, bottom_contour). Contours may be None if not present.
    """

    height, width = bw.shape
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print("height: {}".format(height))

    top_contour = None
    bottom_contour = None

    rectangles = list(map(lambda contour: cv2.boundingRect(contour), contours))

    # Filter top and bottom rect.
    for i in range(0, len(rectangles)):
        rect = rectangles[i]

        # Background must either scratch right or left border
        if rect[0] <= 0 or rect[0] >= width-1:
            if rect[1] == 0:
                top_contour = contours[i]
            elif rect[1]+rect[3] == height:
                bottom_contour = contours[i]

        if top_contour is not None and bottom_contour is not None:
            break

    return top_contour, bottom_contour


def get_background_contour_bottom_line(bw, contour):
    """
    Retrieves contours bottom points and creates line out of it.

    Only works properly if the contour has only one bottom line.

    :param bw: b/w image where the contour comes from. Used for dimensions.
    :param contour: Contour to get bottom line for.
    :return: Tuple (theta in radians, y displacement).
    """
    height, width = bw.shape
    bottom_points = []

    # Just a straight line
    if len(contour) == 2:
        bottom_points.append(contour[0][0])
        bottom_points.append(contour[1][0])

    for p in contour:
        point = p[0]

        # check if inside
        if point[1] > 0:
            bottom_points.append((point[0], point[1]))

    return points_to_line(bottom_points)


def get_background_contour_top_line(bw, contour):
    """
    Retrieves contours top points and creates line out of it.

    Only works properly if the contour has only one bottom line.

    :param bw: b/w image where the contour comes from. Used for dimensions.
    :param contour: Contour to get bottom line for.
    :return: Tuple (theta in radians, y displacement).
    """
    height, width = bw.shape
    bottom_points = []

    # Just a straight line
    if len(contour) == 2:
        bottom_points.append(contour[0][0])
        bottom_points.append(contour[1][0])

    for p in contour:
        point = p[0]

        # check if inside
        if point[1] < height-1:
            bottom_points.append((point[0], point[1]))

    return points_to_line(bottom_points)







