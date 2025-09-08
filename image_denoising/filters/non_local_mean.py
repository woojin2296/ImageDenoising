"""Non-local means filter implementation."""

import numpy as np
from numba import jit


@jit(nopython=True)
def non_local_mean_filter(img, window_size, patch_size, sigma, verbose=True):
    """Apply a non-local means filter to ``img``."""
    height, width, chan = img.shape

    padwidth = window_size // 2
    patch_rad = patch_size // 2

    padded_image = np.zeros((height + window_size, width + window_size, chan), dtype=np.uint8)
    padded_image[padwidth:padwidth + height, padwidth:padwidth + width, :] = img

    output_image = padded_image.copy()

    if verbose:
        iterator = 0
        total_iterations = height * width * (window_size - patch_size) ** 2
        print("TOTAL ITERATIONS =", total_iterations)

    for h in range(padwidth, padwidth + height):
        for w in range(padwidth, padwidth + width):
            winw = w - padwidth
            winh = h - padwidth

            pixel_color = np.zeros(chan)
            total_weight = 0

            original_patch = padded_image[h - patch_rad:h + patch_rad + 1, w - patch_rad:w + patch_rad + 1, :]

            for patchh in range(winh, winh + window_size - patch_size):
                for patchw in range(winw, winw + window_size - patch_size):
                    comp_patch = padded_image[patchh:patchh + patch_size + 1, patchw:patchw + patch_size + 1, :]

                    euclidean_distance = np.sqrt(np.sum((comp_patch - original_patch) ** 2))
                    weight = np.exp(-euclidean_distance / (sigma ** 2) / 2)
                    total_weight += weight

                    pixel_color += weight * padded_image[patchh + patch_rad, patchw + patch_rad, :]
                    iterator += 1

                    if verbose and iterator % 1000000 == 0:
                        percent_complete = iterator * 100 / total_iterations
                        print("COMPLETE =", round(percent_complete, 5), "%")

            pixel_color /= total_weight
            output_image[h, w, :] = pixel_color

    return output_image[padwidth:padwidth + height, padwidth:padwidth + width, :]

