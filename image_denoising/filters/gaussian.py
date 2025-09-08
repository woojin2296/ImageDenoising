"""Gaussian filter implementation using OpenCV."""

import cv2


def gaussian_filter_opencv(image, kernel_size, sigma):
    """Apply a Gaussian blur to ``image``.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    kernel_size : tuple[int, int]
        Gaussian kernel size ``(width, height)``.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    numpy.ndarray
        Blurred image.
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

