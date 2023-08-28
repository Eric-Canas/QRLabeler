import cv2
import numpy as np


def sort_corners(corners: np.ndarray) -> np.ndarray:
    """
    Sort the corners of a quadrilateral in a clockwise order.

    Parameters:
    - corners: 2D numpy array with shape (4, 2) representing the corners of a quadrilateral.

    Returns:
    - A 2D numpy array with shape (4, 2) representing the sorted corners.
    """

    # Compute the centroid of the corners
    centroid = np.mean(corners, axis=0)

    # Compute the angles for each point
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])

    # Sort corners by angles
    sort_order = np.argsort(angles)
    sorted_corners = corners[sort_order]

    return sorted_corners


def select_polygon_corners(image: np.ndarray, prev_corners: np.ndarray = None) -> np.ndarray:
    current_display = image.copy()

    if prev_corners is not None:
        for point in (prev_corners * (image.shape[1], image.shape[0])).astype(int):
            cv2.circle(current_display, tuple(point), 5, (255, 0, 0), -1, cv2.LINE_AA, shift=0)

    corners = []

    def draw_x(img, point, color=(193, 182, 255), size=5):
        cv2.line(img, (point[0] - size, point[1] - size), (point[0] + size, point[1] + size), color, 2, cv2.LINE_AA)
        cv2.line(img, (point[0] + size, point[1] - size), (point[0] - size, point[1] + size), color, 2, cv2.LINE_AA)

    def draw_polygon(img, corners):
        for i in range(len(corners) - 1):
            cv2.line(img, corners[i], corners[i + 1], color=(0, 64, 255), thickness=2, lineType=cv2.LINE_AA)
        for corner in corners:
            draw_x(img, corner)

    def select_point(event, x, y, flags, param):
        nonlocal current_display

        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))
            draw_polygon(current_display, corners)
            cv2.imshow('Image', current_display)

        elif event == cv2.EVENT_MOUSEMOVE and corners:
            temp_img = current_display.copy()
            overlay = current_display.copy()

            # Draw a line from the last point to the cursor
            cv2.line(overlay, corners[-1], (x, y), color=(0, 96, 255), thickness=2, lineType=cv2.LINE_AA)

            # If there are 3 points, draw another line from the first point to the cursor to preview the closed polygon
            if len(corners) == 3:
                cv2.line(overlay, corners[0], (x, y), color=(0, 96, 255), thickness=2, lineType=cv2.LINE_AA)

            cv2.addWeighted(overlay, 0.6, temp_img, 1 - 0.6, 0, temp_img)
            cv2.imshow('Image', temp_img)

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', select_point)

    cv2.imshow('Image', current_display)

    while len(corners) < 4:
        key = cv2.waitKey(1)
        if key & 0xFF == 27:  # the Escape key
            cv2.destroyAllWindows()
            return None
        elif key in (ord('n'), ord('N')):
            cv2.destroyAllWindows()
            return None
        elif key in (ord('d'), ord('D')) and corners:
            corners.pop()
            current_display = image.copy()
            if prev_corners is not None:
                for point in (prev_corners * (image.shape[1], image.shape[0])).astype(int):
                    cv2.circle(current_display, tuple(point), 5, (255, 0, 0), -1, cv2.LINE_AA, shift=0)
            draw_polygon(current_display, corners)
            cv2.imshow('Image', current_display)

    corners = np.array(corners, dtype=np.float32) / (image.shape[1], image.shape[0])
    cv2.destroyAllWindows()

    return corners

