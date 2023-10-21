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
    zoom_factor = 1.0
    pan_offset = (0, 0)
    crop_offset = (0, 0)
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

    def zoom_image(img, factor, center):
        height, width = img.shape[:2]
        new_width = int(width / factor)
        new_height = int(height / factor)

        top_left_x = max(int(center[0] - new_width // 2), 0)
        top_left_y = max(int(center[1] - new_height // 2), 0)

        bottom_right_x = min(top_left_x + new_width, width)
        bottom_right_y = min(top_left_y + new_height, height)

        crop_img = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        zoomed = cv2.resize(crop_img, (width, height), interpolation=cv2.INTER_LINEAR)
        return zoomed, (top_left_x, top_left_y)

    def zoom_and_pan(x, y, factor):
        nonlocal current_display, zoom_factor, crop_offset
        zoom_factor *= factor
        current_display, crop_offset = zoom_image(image, zoom_factor, (x, y))
        cv2.imshow('Image', current_display)

    def select_point(event, x, y, flags, param):
        nonlocal current_display, zoom_factor, pan_offset
        x_original = int((x - pan_offset[1]) / zoom_factor)
        y_original = int((y - pan_offset[0]) / zoom_factor)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Undo the transformation to save the original coordinates
            corners.append((x_original, y_original))
            draw_polygon(current_display,
                         [(int(cx * zoom_factor + pan_offset[1]), int(cy * zoom_factor + pan_offset[0])) for cx, cy in
                          corners])
            cv2.imshow('Image', current_display)

        elif event == cv2.EVENT_RBUTTONDOWN:
            zoom_and_pan(x, y, 4)

        elif event == cv2.EVENT_MOUSEMOVE and corners:

            temp_img = current_display.copy()

            overlay = current_display.copy()

            # Calculate the original coordinates of the last corner and apply zoom and crop offset

            last_corner_x = int((corners[-1][0]) * zoom_factor)

            last_corner_y = int((corners[-1][1]) * zoom_factor)

            # Draw a line from the last corner to the current mouse position

            cv2.line(overlay, (last_corner_x, last_corner_y), (x, y), color=(0, 96, 255), thickness=2,
                     lineType=cv2.LINE_AA)

            if len(corners) >= 3:
                # Calculate the original coordinates of the first corner and apply zoom and crop offset

                first_corner_x = int((corners[0][0]) * zoom_factor)

                first_corner_y = int((corners[0][1]) * zoom_factor)

                # Draw a line from the first corner to the current mouse position

                cv2.line(overlay, (first_corner_x, first_corner_y), (x, y), color=(0, 96, 255), thickness=2,
                         lineType=cv2.LINE_AA)

            cv2.addWeighted(overlay, 0.6, temp_img, 1 - 0.6, 0, temp_img)

            cv2.imshow('Image', temp_img)

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', select_point)
    cv2.imshow('Image', current_display)

    while True:
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
            draw_polygon(current_display, corners)
            cv2.imshow('Image', current_display)
        elif key in (ord('y'), ord('Y')):
            if len(corners) >= 4:
                # Convert to a NumPy array for convenience
                corners = np.array(corners, dtype=np.float32)

                # Reverse the zoom and crop offset
                corners += crop_offset
                # Convert to normalized coordinates ([0, 1]).
                corners /= np.array([image.shape[1], image.shape[0]])
                cv2.destroyAllWindows()
                return corners
            else:
                print("You cannot accept polygons with less than 4 points. Click N to discard image.")



