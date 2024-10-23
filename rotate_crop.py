import os
import math
import cv2
import numpy as np
from deskew import determine_skew
from typing import Tuple, Union


def rotate(
    image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Rotates an image by a specified angle while preserving its content.

    Parameters:
    - image (np.ndarray): The input image to be rotated.
    - angle (float): The angle in degrees by which to rotate the image.
    - background (Union[int, Tuple[int, int, int]]): The color of the background to fill
      in the area created by the rotation. This can be a single integer (for grayscale)
      or a tuple of three integers (for RGB).

    Returns:
    - np.ndarray: The rotated image with the specified background color.
    """
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(
        np.cos(angle_radian) * old_width
    )
    height = abs(np.sin(angle_radian) * old_width) + abs(
        np.cos(angle_radian) * old_height
    )
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(
        image, rot_mat, (int(round(height)), int(round(width))), borderValue=background
    )


def rotate_crop():
    """
    Rotates and crops images from the input folder based on the presence of magenta colors.

    This function performs the following steps:
    1. Detects and lists all `.tif` and `.png` files in the 'input/raw' directory.
    2. Reads each image and thresholds it to isolate magenta regions.
    3. Determines the skew angle of the image and rotates it accordingly.
    4. Computes cropping coordinates based on the detected magenta areas.
    5. Crops the rotated image and saves the result in the 'output/crop' directory.

    Returns:
    None
    """
    files = []  # detect .tif files in input folder
    for file in os.listdir(os.path.join("input", "raw")):
        if file.endswith(".tif"):
            files += [file]
        if file.endswith(".png"):
            files += [file]

    for file in files:
        img = cv2.imread(os.path.join("input", "raw", file))  # read image in loop

        # threshold magenta
        lower = np.array([100, 0, 100])
        upper = np.array([255, 80, 255])
        thresh = cv2.inRange(img, lower, upper)

        # Change non-magenta to white
        result = img.copy()
        result[thresh != 255] = (255, 255, 255)

        angle = determine_skew(result, angle_pm_90=True)
        rotated = rotate(img, angle, (0, 0, 0))

        print(f"The angle in {file} is {angle} degrees.")

        thresh = cv2.inRange(rotated, lower, upper)

        # https://stackoverflow.com/questions/68529412/how-to-crop-images-based-on-mask-threshold
        xsize_half = 700 / 2
        ysize_half = 2300 / 2
        xx, yy = thresh.nonzero()
        if int(xx.mean()) <= ysize_half:
            range_h = ysize_half
        if int(xx.mean()) > ysize_half:
            range_h = int(xx.mean())
        if int(yy.mean()) <= xsize_half:
            range_v = xsize_half
        if int(yy.mean()) > xsize_half:
            range_v = int(yy.mean())
        max_crop_h = int(range_h + ysize_half)
        min_crop_h = int(range_h - ysize_half)
        max_crop_v = int(range_v + xsize_half)
        min_crop_v = int(range_v - xsize_half)

        min_crop_v = max(min_crop_v, 0)
        min_crop_h = max(min_crop_h, 0)

        crop = rotated[min_crop_h:max_crop_h, min_crop_v:max_crop_v]

        # save results
        cv2.imwrite(os.path.join("output", "crop", file), crop)
