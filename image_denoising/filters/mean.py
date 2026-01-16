"""Mean filter implementation using OpenCV."""

import cv2


def mean_filter_opencv(img, filter_size):
    """Apply a mean filter to ``img`` using OpenCV."""
    kernel_size = filter_size // 2 + 1
    return cv2.blur(img, (kernel_size, kernel_size))

