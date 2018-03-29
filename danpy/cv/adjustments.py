from __future__ import division

import numpy as np
import cv2


def gammaCorrection(image, gamma=1.0):
    """Applies gamma correction to input image
    Args:
        image: input image as numpy array
        gamma: value of gamma; default is 1
    Returns:
        gamma corrected image
    """
    # Check if gamma value is valid
    if gamma < 0:
        raise Exception("Invalid Gamma value: " + gamma)

    # Return image if no gamma value specified or equals 1
    if gamma == 1.0:
        return image
    
    # Create look-up table
    table = np.array([((i/255.0)**gamma)*255
        for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction using look-up table
    return cv2.LUT(image, table)


def histEqualize(image):
    """Applies Histogram Equalization to both grayscale and
       BGR color image
    Args:
        image: input image
    Returns:
        histogram equalized image
    """
    # Apply inbuilt opencv histogram equalizer for grayscale image
    if image.shape[2] == 1:
        return cv2.equalizeHist(image)
    
    # Raise error if more than 3 channels
    if not image.shape[2] == 3:
        raise Exception("Expected image with 3 channels.")

    # Convert from BGR to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Equalize histogram of Luminance channel only
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

    # Convert back to BGR color space
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
