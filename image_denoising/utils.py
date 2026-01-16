"""Utility helpers for image loading and evaluation."""

import os
from skimage import io, metrics


def calculate_psnr(original, denoised):
    """Calculate the PSNR between two images."""
    return metrics.peak_signal_noise_ratio(original, denoised)


def load_images(path):
    """Load and return all PNG images within ``path``.

    Parameters
    ----------
    path : str
        Directory containing PNG files.

    Returns
    -------
    tuple[list, list]
        A tuple of the loaded images and their file names.
    """
    images = []
    image_names = []

    for filename in sorted(os.listdir(path)):
        if filename.endswith(".png"):
            img_path = os.path.join(path, filename)
            img = io.imread(img_path)
            images.append(img)
            image_names.append(filename)

    return images, image_names

