from statistics import mean

import cv2

from util import points_to_line, contour_center, calc_line


def get_background_contours(bw):
    """
    Tries to retrieve the areas above and below the negative strip (the background).
    If not found, None will be returned.

    :param bw: Strip as b/w image. Strip must be dark, sprocket holes and "background" white.
    :return: Tuple (top_contour, bottom_contour). Contours may be None if not present.
    """

    height, width = bw.shape
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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


def get_sprocket_holes_contours(bw, top_bg_line, bottom_bg_line):
    """
    Finds all sprocket holes. Uses top and bottom background border line to
    evaluate if contour is within the negative strip. Also used to determine middle
    of negative strip to decide of contours is top or bottom hole.

    If either top or bottom line is None 0 or height-1 are used as top and bottom
    border lines.

    :param bw: b/w negative strip image. Strip must be black, holes and background white.
    :param top_bg_line: Border line of top background. Can be None.
    :param bottom_bg_line:  Border line of bottom background. Can be None.
    :return: Tuple (top_holes, bottom_holes). Each value is a list of contours for the sprocket holes.
    """
    height, width = bw.shape

    all_contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Now lets filter out all the contours that are above or below the given lines
    contours_top = []
    contours_bottom = []

    for contour in all_contours:
        cx, cy = contour_center(contour)

        # y for cx for the top line, if line is None 0
        top_bg_y = 0 if top_bg_line is None else calc_line(cx, top_bg_line)

        # y for cx for the bottom line, if line is None bottom line
        bottom_bg_y = height-1 if top_bg_line is None else calc_line(cx, bottom_bg_line)

        # between top and bottom
        middle_y = top_bg_y + 0.5 * (bottom_bg_y-top_bg_y)

        # top sprocket hole
        if top_bg_y < cy < middle_y:
            contours_top.append(contour)

        # bottom sprocket hole
        elif middle_y < cy < bottom_bg_y:
            contours_bottom.append(contour)

    return contours_top, contours_bottom


def get_background_border_lines(bw):
    """
    Retrieves the border lines between background and negative strip.

    If top or bottom background is not visible, the corresponding line entry
    in the return tuple will just a straight line at the top or bottom.

    :param bw: b/w image, negative strip must be black and the background white.
    :return: Tuple (top_line, bottom_line). Lines are represented by (theta, displacement)
    """
    height, width = bw.shape
    (top, bottom) = get_background_contours(bw)

    top_line = (0, 0) if top is None else get_background_contour_bottom_line(bw, top)
    bottom_line = (0, height-1) if bottom is None else get_background_contour_top_line(bw, bottom)

    return top_line, bottom_line


def get_sprocket_holes_lines(holes_contours):
    """
    Retrieves top and bottom line of all holes.
    At least two holes must be given.

    To work properly, the holes must lay nearly on one line.

    :param holes_contours: Contours of all sprocket holes.
    :return: Tuple (top_line, bottom_line). Each line is represented by (theta, displacement).
    """

    assert len(holes_contours) > 1, "Need at least two holes"

    # Create list of tuples. Each tuple has (contour, (cx, cy))
    # List will be sorted by cx. Center is only used for sorting.
    holes = sorted(
        list(map(lambda contour: (contour, contour_center(contour)), holes_contours)),
        key=lambda t: t[1][0]
    )

    # Now lets grab all top and all bottom points
    top_points = []
    bottom_points = []

    for hole in holes:
        contour = hole[0]

        # todo: understand, how this works!
        # https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
        extreme_top = tuple(contour[contour[:, :, 1].argmin()][0])
        extreme_bottom = tuple(contour[contour[:, :, 1].argmax()][0])

        top_points.append(extreme_top)
        bottom_points.append(extreme_bottom)

    top_line = points_to_line(top_points)
    bottom_line = points_to_line(bottom_points)

    return top_line, bottom_line
