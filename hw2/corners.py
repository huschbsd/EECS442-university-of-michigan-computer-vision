import os

import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.

from common import read_img, save_img


def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    H = image.shape[0]
    W = image.shape[1]
    padding_vertical = (window_size[0]//2, window_size[0]//2)
    padding_horizon = (window_size[1]//2, window_size[1]//2)

    #offsetting the image by (u, v);
    image_shifted = np.roll(image, u, axis=0)
    image_shifted = np.roll(image_shifted, v, axis=1)

    #taking the squared difference with the original image
    square_diff = np.square(image_shifted - image)

    ##summing up the values within a window using convolution with zero-padding
    window = np.ones(window_size)
    output = scipy.ndimage.convolve(square_diff, window, mode='constant', cval=0.0)

    return output

def gaussian_kernel_generator(sigma, size=(3,3)):
    output = np.zeros(size)
    H = size[0]
    W = size[1]

    m = H//2
    n = W//2

    for i in range(-m, m+1):
        for j in range(-n, n+1):
            output[i+m,j+n] = (1.0/(2*(np.pi)*(sigma**2)))*np.exp(-((i**2) + (j**2))/(2*(sigma**2)))

    #smoothing filter
    output_sum = np.sum(output)
    output = output/output_sum
    return output

def harris_detector(image, window_size=(5, 5)):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: a image of size H x W
    """
    # compute the derivatives
    H = image.shape[0]
    W = image.shape[1]

    kx = np.array([[1, 0, -1]])
    ky = np.array([[1], [0], [-1]])

    Ix = scipy.ndimage.convolve(image, kx, mode='constant', cval=0.0)
    Iy = scipy.ndimage.convolve(image, ky, mode='constant', cval=0.0)

    window = gaussian_kernel_generator(1, window_size)
    
    #build matrix M
    Ixx = scipy.ndimage.convolve(((Ix)**2), window, mode='constant', cval=0.0)
    Iyy = scipy.ndimage.convolve(((Iy)**2), window, mode='constant', cval=0.0)
    #Ixx = scipy.ndimage.convolve(Ix.dot(Ix), window, mode='constant', cval=0.0)
    Ixy = scipy.ndimage.convolve((Ix * Iy), window, mode='constant', cval=0.0)
    M = np.zeros((H,W,2,2))

    for i in range(H):
        for j in range(W):
            M[i,j,0,0] = Ixx[i,j]
            M[i,j,0,1] = Ixy[i,j]
            M[i,j,1,0] = Ixy[i,j]
            M[i,j,1,1] = Iyy[i,j]

    # For each image location, construct the structure tensor and calculate
    # the Harris response
    response = np.zeros((H,W))
    alpha = 0.05

    for i in range(H):
        for j in range(W):
            response[i,j] = np.linalg.det(M[i,j]) - alpha*((np.trace(M[i,j]))**2)

    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # -- TODO Task 5: Corner Score --
    # (a): Complete corner_score()

    # (b)
    # Define offsets and window size and calulcate corner score
    u, v, W = 5, 0, (5,5)

    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score.png")

    # (c): No Code

    # -- TODO Task 6: Harris Corner Detector --
    # (a): Complete harris_detector()

    # (b)
    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")


if __name__ == "__main__":
    main()
