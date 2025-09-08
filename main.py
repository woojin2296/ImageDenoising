"""Command line interface for image denoising filters."""

import argparse
from skimage import io

from image_denoising import (
    gaussian_filter_opencv,
    hybrid_median_filter,
    mean_filter_opencv,
    median_filter,
    median_filter_opencv,
    non_local_mean_filter,
    load_images,
)

FILTERS = {
    "gaussian": lambda img, args: gaussian_filter_opencv(
        img, (args.kernel_size, args.kernel_size), args.sigma
    ),
    "mean": lambda img, args: mean_filter_opencv(img, args.kernel_size),
    "median": lambda img, args: median_filter_opencv(img, args.kernel_size),
    "median_naive": lambda img, args: median_filter(img, args.kernel_size),
    "hybrid_median": lambda img, args: hybrid_median_filter(img, args.kernel_size),
    "non_local_mean": lambda img, args: non_local_mean_filter(
        img, args.window_size, args.patch_size, args.sigma
    ),
}


def apply_filter(args):
    """Load an image, apply the chosen filter, and save the result."""
    img = io.imread(args.input)
    result = FILTERS[args.filter](img, args)
    io.imsave(args.output, result)


def run_nlm_grid_search():
    """Replicate the original non-local means grid search experiment."""
    img = io.imread("image/set1/noisy/dog_noisy.png")
    for window_size in range(5, 30, 5):
        for patch_size in range(2, 12, 2):
            if window_size <= patch_size:
                continue
            for sigma in range(25, 50, 5):
                result = non_local_mean_filter(img, window_size, patch_size, sigma)
                io.imsave(
                    f"image/set1/result/dog_nlm_{window_size}_{patch_size}_{sigma}.png",
                    result,
                )

    images, names = load_images("image/set2/result/median/")
    for image, name in zip(images, names):
        for window_size in range(15, 20, 5):
            for patch_size in range(2, 4, 2):
                if window_size <= patch_size:
                    continue
                for sigma in range(10, 15, 5):
                    result = non_local_mean_filter(image, window_size, patch_size, sigma)
                    io.imsave(
                        f"image/set1/result/{name.replace('.png', '_')}{window_size}_{patch_size}_{sigma}.png",
                        result,
                    )

    img = io.imread("image/set2/noisy/card_noisy.png")
    result = non_local_mean_filter(img, 15, 2, 10)
    io.imsave("image/set2/result/nlm/card_nlm_15_2_10.png", result)


def build_parser():
    parser = argparse.ArgumentParser(description="Image denoising filters CLI")
    sub = parser.add_subparsers(dest="command")

    apply_p = sub.add_parser("apply", help="Apply a single filter to an image")
    apply_p.add_argument("filter", choices=list(FILTERS.keys()))
    apply_p.add_argument("input", help="Path to input image")
    apply_p.add_argument("output", help="Path to save the filtered image")
    apply_p.add_argument(
        "--kernel-size", type=int, default=3, help="Kernel size for relevant filters"
    )
    apply_p.add_argument("--sigma", type=float, default=1.0, help="Sigma value")
    apply_p.add_argument(
        "--window-size", type=int, default=10, help="Window size for non-local means"
    )
    apply_p.add_argument(
        "--patch-size", type=int, default=5, help="Patch size for non-local means"
    )

    sub.add_parser("nlm-grid", help="Run the non-local means grid search demo")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "apply":
        apply_filter(args)
    elif args.command == "nlm-grid":
        run_nlm_grid_search()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

