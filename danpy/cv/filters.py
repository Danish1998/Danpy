import numpy as np
import cv2


def nothing(x):
    """Empty function to pass when function type is required"""
    pass


def HSV_Picker(hsv_image, output="HSV Mask"):
    """Displays sliders to find the optimum threshold values
    Args:
        hsv_image: input image in HSV color space
        output: name of the window used to display the output mask
    Returns:
        thresh_low: lower bound of threshold values as numpy array
        thresh_high: upper bound of threshold values as numpy array
        hsv_mask: binary image of resultant mask
    """
    # Maximum values for each channel
    MAX_H_VALUE = 255
    MAX_S_VALUE = 255
    MAX_V_VALUE = 255
    
    # Low & high thresholds
    thresh_low = [0, 0, 0]
    thresh_high = [MAX_H_VALUE, MAX_S_VALUE, MAX_V_VALUE]
    
    # Create hsv mask
    sz = (hsv_image.shape[0], hsv_image.shape[1])
    hsv_mask = np.zeros(sz)
    
    # Create the window for trackbars and output
    cv2.namedWindow("Thresholds", cv2.WINDOW_NORMAL)
    cv2.namedWindow(output)
    
    # Create trackbars for color change
    cv2.createTrackbar(" H low ", "Thresholds", 0, MAX_H_VALUE, nothing)
    cv2.createTrackbar(" H high ", "Thresholds", MAX_H_VALUE, MAX_H_VALUE, nothing)
    cv2.createTrackbar(" S low ", "Thresholds", 0, MAX_S_VALUE, nothing)
    cv2.createTrackbar(" S high ", "Thresholds", MAX_S_VALUE, MAX_S_VALUE, nothing)
    cv2.createTrackbar(" V low ", "Thresholds", 0, MAX_V_VALUE, nothing)
    cv2.createTrackbar(" V high ", "Thresholds", MAX_V_VALUE, MAX_V_VALUE, nothing)
    
    # Create trackbars for kernel size & iterations
    cv2.createTrackbar(" KSize ", "Thresholds", 5, 35, nothing)
    cv2.createTrackbar(" Iter ", "Thresholds", 1, 10, nothing)
    
    while(True):
        # Get trackbar positions
        h_low = cv2.getTrackbarPos(" H low ", "Thresholds")
        h_high = cv2.getTrackbarPos(" H high ", "Thresholds")
        s_low = cv2.getTrackbarPos(" S low ", "Thresholds")
        s_high = cv2.getTrackbarPos(" S high ", "Thresholds")
        v_low = cv2.getTrackbarPos(" V low ", "Thresholds")
        v_high = cv2.getTrackbarPos(" V high ", "Thresholds")
        open_ksize = max(cv2.getTrackbarPos(" KSize ", "Thresholds"), 1)
        open_iter = max(cv2.getTrackbarPos(" Iter ", "Thresholds"), 1)

        # Set threshold values
        thresh_low = np.array([h_low, s_low, v_low])
        thresh_high = np.array([h_high, s_high, v_high])

        # Get hsv mask
        hsv_mask = cv2.inRange(hsv_image, thresh_low, thresh_high)
        
        # Apply morph opening to reduce noise
        kernel = np.ones((open_ksize, open_ksize), np.uint8)
        for i in xrange(0, open_iter):
            hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
        
        # Display hsv mask
        cv2.imshow(output, hsv_mask)
        
        if cv2.waitKey(1) & 0xff == 27:
            cv2.destroyWindow("Thresholds")
            break
            
    return thresh_low, thresh_high, hsv_mask


def RGB_Picker(rgb_image, output="RGB Mask"):
    """Displays sliders to find the optimum threshold values
    Args:
        rgb_image: input image in RGB color space
        output: name of the window used to display the output mask
    Returns:
        thresh_low: lower bound of threshold values as numpy array
        thresh_high: upper bound of threshold values as numpy array
        rgb_mask: binary image of resultant mask
    """
    # Maximum values for each channel
    MAX_R_VALUE = 255
    MAX_G_VALUE = 255
    MAX_B_VALUE = 255

    # Low & high thresholds
    thresh_low = [0, 0, 0]
    thresh_high = [MAX_R_VALUE, MAX_G_VALUE, MAX_B_VALUE]

    # Create rgb mask
    sz = (rgb_image.shape[0], rgb_image.shape[1])
    rgb_mask = np.zeros(sz)

    # Create the window for trackbars and output
    cv2.namedWindow("Thresholds", cv2.WINDOW_NORMAL)
    cv2.namedWindow(output)

    # Create trackbars for color change
    cv2.createTrackbar(" R low ", "Thresholds", 0, MAX_R_VALUE, nothing)
    cv2.createTrackbar(" R high ", "Thresholds", MAX_R_VALUE, MAX_R_VALUE, nothing)
    cv2.createTrackbar(" G low ", "Thresholds", 0, MAX_G_VALUE, nothing)
    cv2.createTrackbar(" G high ", "Thresholds", MAX_G_VALUE, MAX_G_VALUE, nothing)
    cv2.createTrackbar(" B low ", "Thresholds", 0, MAX_B_VALUE, nothing)
    cv2.createTrackbar(" B high ", "Thresholds", MAX_B_VALUE, MAX_B_VALUE, nothing)

    # Create trackbars for kernel size & iterations
    cv2.createTrackbar(" KSize ", "Thresholds", 5, 35, nothing)
    cv2.createTrackbar(" Iter ", "Thresholds", 1, 10, nothing)

    while (True):
        # Get trackbar positions
        r_low = cv2.getTrackbarPos(" R low ", "Thresholds")
        r_high = cv2.getTrackbarPos(" R high ", "Thresholds")
        g_low = cv2.getTrackbarPos(" G low ", "Thresholds")
        g_high = cv2.getTrackbarPos(" G high ", "Thresholds")
        b_low = cv2.getTrackbarPos(" B low ", "Thresholds")
        b_high = cv2.getTrackbarPos(" B high ", "Thresholds")
        open_ksize = max(cv2.getTrackbarPos(" KSize ", "Thresholds"), 1)
        open_iter = max(cv2.getTrackbarPos(" Iter ", "Thresholds"), 1)

        # Set threshold values
        thresh_low = np.array([r_low, g_low, b_low])
        thresh_high = np.array([r_high, g_high, b_high])

        # Get rgb mask
        rgb_mask = cv2.inRange(rgb_image, thresh_low, thresh_high)

        # Apply morph opening to reduce noise
        kernel = np.ones((open_ksize, open_ksize), np.uint8)
        for i in xrange(0, open_iter):
            rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_OPEN, kernel)

        # Display rgb mask
        cv2.imshow(output, rgb_mask)

        if cv2.waitKey(1) & 0xff == 27:
            cv2.destroyWindow("Thresholds")
            break

    return thresh_low, thresh_high, rgb_mask


def Lab_Picker(Lab_image, output="Lab Mask"):
    """Displays sliders to find the optimum threshold values
    Args:
        Lab_image: input image in Lab color space
        output: name of the window used to display the output mask
    Returns:
        thresh_low: lower bound of threshold values as numpy array
        thresh_high: upper bound of threshold values as numpy array
        Lab_mask: binary image of resultant mask
    """
    # Maximum values for each channel
    MAX_L_VALUE = 255
    MAX_a_VALUE = 255
    MAX_b_VALUE = 255

    # Low & high thresholds
    thresh_low = [0, 0, 0]
    thresh_high = [MAX_L_VALUE, MAX_a_VALUE, MAX_b_VALUE]

    # Create Lab mask
    sz = (Lab_image.shape[0], Lab_image.shape[1])
    Lab_mask = np.zeros(sz)

    # Create the window for trackbars and output
    cv2.namedWindow("Thresholds", cv2.WINDOW_NORMAL)
    cv2.namedWindow(output)

    # Create trackbars for color change
    cv2.createTrackbar(" L low ", "Thresholds", 0, MAX_L_VALUE, nothing)
    cv2.createTrackbar(" L high ", "Thresholds", MAX_L_VALUE, MAX_L_VALUE, nothing)
    cv2.createTrackbar(" a low ", "Thresholds", 0, MAX_a_VALUE, nothing)
    cv2.createTrackbar(" a high ", "Thresholds", MAX_a_VALUE, MAX_a_VALUE, nothing)
    cv2.createTrackbar(" b low ", "Thresholds", 0, MAX_b_VALUE, nothing)
    cv2.createTrackbar(" b high ", "Thresholds", MAX_b_VALUE, MAX_b_VALUE, nothing)

    # Create trackbars for kernel size & iterations
    cv2.createTrackbar(" KSize ", "Thresholds", 5, 35, nothing)
    cv2.createTrackbar(" Iter ", "Thresholds", 1, 10, nothing)

    while (True):
        # Get trackbar positions
        L_low = cv2.getTrackbarPos(" L low ", "Thresholds")
        L_high = cv2.getTrackbarPos(" L high ", "Thresholds")
        a_low = cv2.getTrackbarPos(" a low ", "Thresholds")
        a_high = cv2.getTrackbarPos(" a high ", "Thresholds")
        b_low = cv2.getTrackbarPos(" b low ", "Thresholds")
        b_high = cv2.getTrackbarPos(" b high ", "Thresholds")
        open_ksize = max(cv2.getTrackbarPos(" KSize ", "Thresholds"), 1)
        open_iter = max(cv2.getTrackbarPos(" Iter ", "Thresholds"), 1)

        # Set threshold values
        thresh_low = np.array([L_low, a_low, b_low])
        thresh_high = np.array([L_high, a_high, b_high])

        # Get Lab mask
        Lab_mask = cv2.inRange(Lab_image, thresh_low, thresh_high)

        # Apply morph opening to reduce noise
        kernel = np.ones((open_ksize, open_ksize), np.uint8)
        for i in xrange(0, open_iter):
            Lab_mask = cv2.morphologyEx(Lab_mask, cv2.MORPH_OPEN, kernel)

        # Display Lab mask
        cv2.imshow(output, Lab_mask)

        if cv2.waitKey(1) & 0xff == 27:
            cv2.destroyWindow("Thresholds")
            break

    return thresh_low, thresh_high, Lab_mask


def YCrCb_Picker(ycrcb_image, output="YCrCb Mask"):
    """Displays sliders to find the optimum threshold values
    Args:
        ycrcb_image: input image in YCrCb color space
        output: name of the window used to display the output mask
    Returns:
        thresh_low: lower bound of threshold values as numpy array
        thresh_high: upper bound of threshold values as numpy array
        ycrcb_mask: binary image of resultant mask
    """
    # Maximum values for each channel
    MAX_Y_VALUE = 255
    MAX_Cr_VALUE = 255
    MAX_Cb_VALUE = 255

    # Low & high thresholds
    thresh_low = [0, 0, 0]
    thresh_high = [MAX_Y_VALUE, MAX_Cr_VALUE, MAX_Cb_VALUE]

    # Create YCrCb mask
    sz = (ycrcb_image.shape[0], ycrcb_image.shape[1])
    ycrcb_mask = np.zeros(sz)

    # Create the window for trackbars and output
    cv2.namedWindow("Thresholds", cv2.WINDOW_NORMAL)
    cv2.namedWindow(output)

    # Create trackbars for color change
    cv2.createTrackbar(" Y low ", "Thresholds", 0, MAX_Y_VALUE, nothing)
    cv2.createTrackbar(" Y high ", "Thresholds", MAX_Y_VALUE, MAX_Y_VALUE, nothing)
    cv2.createTrackbar(" Cr low ", "Thresholds", 0, MAX_Cr_VALUE, nothing)
    cv2.createTrackbar(" Cr high ", "Thresholds", MAX_Cr_VALUE, MAX_Cr_VALUE, nothing)
    cv2.createTrackbar(" Cb low ", "Thresholds", 0, MAX_Cb_VALUE, nothing)
    cv2.createTrackbar(" Cb high ", "Thresholds", MAX_Cb_VALUE, MAX_Cb_VALUE, nothing)

    # Create trackbars for kernel size & iterations
    cv2.createTrackbar(" KSize ", "Thresholds", 5, 35, nothing)
    cv2.createTrackbar(" Iter ", "Thresholds", 1, 10, nothing)

    while (True):
        # Get trackbar positions
        y_low = cv2.getTrackbarPos(" Y low ", "Thresholds")
        y_high = cv2.getTrackbarPos(" Y high ", "Thresholds")
        cr_low = cv2.getTrackbarPos(" Cr low ", "Thresholds")
        cr_high = cv2.getTrackbarPos(" Cr high ", "Thresholds")
        cb_low = cv2.getTrackbarPos(" Cb low ", "Thresholds")
        cb_high = cv2.getTrackbarPos(" Cb high ", "Thresholds")
        open_ksize = max(cv2.getTrackbarPos(" KSize ", "Thresholds"), 1)
        open_iter = max(cv2.getTrackbarPos(" Iter ", "Thresholds"), 1)

        # Set threshold values
        thresh_low = np.array([y_low, cr_low, cb_low])
        thresh_high = np.array([y_high, cr_high, cb_high])

        # Get YCrCb mask
        ycrcb_mask = cv2.inRange(ycrcb_image, thresh_low, thresh_high)

        # Apply morph opening to reduce noise
        kernel = np.ones((open_ksize, open_ksize), np.uint8)
        for i in xrange(0, open_iter):
            ycrcb_mask = cv2.morphologyEx(ycrcb_mask, cv2.MORPH_OPEN, kernel)

        # Display YCrCb mask
        cv2.imshow(output, ycrcb_mask)

        if cv2.waitKey(1) & 0xff == 27:
            cv2.destroyWindow("Thresholds")
            break

    return thresh_low, thresh_high, ycrcb_mask


def colorPicker(image, color_space=None, output=None):
    """Displays sliders to find the optimum threshold values in the specified color space
    Args:
        image: input image in specified color channels
        color_space: color space of input image
        output: name of the window used to display the output mask
    Returns:
        thresh_low: lower bound of threshold values as numpy array
        thresh_high: upper bound of threshold values as numpy array
        mask: binary image of resultant mask
    """
    # Check if the image has 3 color channels
    if not image.shape[2] == 3:
        raise Exception("Image containing 3 channels expected, but contains {} channels.".format(image.shape[2]))

    # List of color spaces for which color picker functions are available
    color_spaces = ["RGB", "HSV", "Lab", "YCrCb"]

    # Default values for thresholds & mask in case of any error
    thresh_low = np.array([0, 0, 0])
    thresh_high = np.array([255, 255, 255])
    mask = np.ones_like(image)

    # Check if color space is provided and valid
    if color_space is None:
        color_space = color_spaces[1]
    if not color_space in color_spaces:
        print("No colorPicker function is available for " + color_space + " color space.")
        return thresh_low, thresh_high, mask

    if color_space == "RGB":
        thresh_low, thresh_high, mask = RGB_Picker(image, output)
    if color_space == "HSV":
        thresh_low, thresh_high, mask = HSV_Picker(image, output)
    if color_space == "Lab":
        thresh_low, thresh_high, mask = Lab_Picker(image, output)
    if color_space == "YCrCb":
        thresh_low, thresh_high, mask = YCrCb_Picker(image, output)

    return thresh_low, thresh_high, mask
