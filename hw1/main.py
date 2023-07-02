"""
Starter code for EECS 442 W21 HW1
"""
import os
import cv2
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from util import generate_gif, renderCube


def rotX(theta):
    """
    Generate 3D rotation matrix about X-axis
    Input:  theta: rotation angle about X-axis
    Output: Rotation matrix (3 x 3 array)
    
    """
    
    
    rot_x = np.zeros((3,3))
    rot_x[0,0] =1
    rot_x[1,1] = np.cos(theta)
    rot_x[1,2] = -(np.sin(theta))
    rot_x[2,1] = np.sin(theta)
    rot_x[2,2] = np.cos(theta) 
    return rot_x


def rotY(theta):
    """
    Generate 3D rotation matrix about Y-axis
    Input:  theta: rotation angle along y-axis
    Output: Rotation matrix (3 x 3 array)
    """
    rot_y = np.zeros((3,3))
    rot_y[0,0] = np.cos(theta)
    rot_y[0,2] = np.sin(theta)
    rot_y[1,1] = 1
    rot_y[2,0] = -(np.sin(theta))
    rot_y[2,2] = np.cos(theta)
    return rot_y


def part1():
    # TODO: Solution for Q1
    # Task 1: Use rotY() to generate cube.gif
    rotList = [rotY(0), rotY(np.pi / 3), rotY(np.pi / 3 * 2), rotY(np.pi), rotY(np.pi / 3 * 4), rotY(np.pi / 3 * 5)]
    generate_gif(rotList, "rotList.gif")
    

    # Task 2:  Use rotX() and rotY() sequentially to check
    # the commutative property of Rotation Matrices
    rotList2 = [rotY(np.pi/4), np.dot(rotX(np.pi/4), rotY(np.pi/4))]
    generate_gif(rotList2, "YX.gif")
    rotList3 = [rotX(np.pi/4), np.dot(rotY(np.pi/4), rotX(np.pi/4))]
    generate_gif(rotList3, "XY.gif")
    
    # Task 3: Combine rotX() and rotY() to render a cube 
    # projection such that end points of diagonal overlap
    # Hint: Try rendering the cube with multiple configrations
    # to narrow down the search region
    rotList4 = [np.dot(rotY(np.arcsin(np.sqrt(3) / 3)), rotX(np.pi/4))]
    generate_gif(rotList4, "onepoint.gif")


def split_triptych(trip):
    """
    Split a triptych into thirds
    Input:  trip: a triptych (H x W matrix)
    Output: R, G, B martices
    """
    row, col = trip.shape
    B, G, R = trip[0 : row // 3, :], trip[row // 3 : row // 3 * 2, :], trip[row // 3 * 2 : , :]
    # TODO: Split a triptych into thirds and 
    # return three channels as numpy arrays
    return R, G, B


def normalized_cross_correlation(ch1, ch2):
    """
    Calculates similarity between 2 color channels
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
    Output: normalized cross correlation (scalar)
    """
    ch1_normed = ch1 / np.linalg.norm(ch1)
    ch2_normed = ch2 / np.linalg.norm(ch2)
    return np.sum(ch1_normed * ch2_normed)


def best_offset(ch1, ch2, metric, Xrange=np.arange(-10, 10), 
                Yrange=np.arange(-10, 10)):
    """
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
            metric: similarity measure between two channels
            Xrange: range to search for optimal offset in vertical direction
            Yrange: range to search for optimal offset in horizontal direction
    Output: optimal offset for X axis and optimal offset for Y axis

    Note: Searching in Xrange would mean moving in the vertical 
    axis of the image/matrix, Yrange is the horizontal axis 
    """
    # TODO: Use metric to align ch2 to ch1 and return optimal offsets
    product = metric(ch1[15:-15,15:-15], ch2[15:-15,15:-15])
    opt_vmov = 0
    opt_hmov = 0
    for vmov in Xrange:
        for hmov in Yrange:
            ch2_roll = np.roll(ch2, vmov, axis=0)
            ch2_roll = np.roll(ch2_roll, hmov, axis=1)
            product_temp = metric(ch1[15:-15, 15:-15], ch2_roll[15:-15, 15:-15])
            if product_temp > product:
                product = product_temp
                opt_hmov = hmov
                opt_vmov = vmov
    return opt_vmov, opt_hmov


def align_and_combine(R, G, B, metric):
    """
    Input:  R: red channel
            G: green channel
            B: blue channel
            metric: similarity measure between two channels
    Output: aligned RGB image 
    """
    # TODO: Use metric to align the three channels 
    # Hint: Use one channel as the anchor to align other two
    opt_G_v, opt_G_h = best_offset(R, G, metric)
    opt_B_v, opt_B_h = best_offset(R, B, metric)
    G = np.roll(G, opt_G_v, axis=0)
    G = np.roll(G, opt_G_h, axis=1)
    B = np.roll(B, opt_B_v, axis=0)
    B = np.roll(B, opt_B_h, axis=1)
    img = np.dstack((R, G, B))
    return img


def pyramid_align(filename):
    # TODO: Reuse the functions from task 2 to perform the 
    # image pyramid alignment iteratively or recursively
    img = plt.imread("tableau/" + filename + ".jpg")
    R, G, B = split_triptych(img)
    row, col = R.shape
    # level-0
    dim_0 = (col // 16, row // 16)
    R_0 = cv2.resize(R, dim_0, interpolation=cv2.INTER_AREA)
    G_0 = cv2.resize(G, dim_0, interpolation=cv2.INTER_AREA)
    B_0 = cv2.resize(B, dim_0, interpolation=cv2.INTER_AREA)
    plt.imsave(filename + "_0_origin.jpg", np.dstack((R_0, G_0, B_0)))
    img = align_and_combine(R_0, G_0, B_0, normalized_cross_correlation)
    plt.imsave(filename + "_0.jpg", img)
    best_G_0 = best_offset(R_0, G_0, normalized_cross_correlation)
    best_B_0 = best_offset(R_0, B_0, normalized_cross_correlation)
    # level-1
    dim_1 = (col // 4, row // 4)
    R_1 = cv2.resize(R, dim_1, interpolation=cv2.INTER_AREA)
    G_1 = cv2.resize(G, dim_1, interpolation=cv2.INTER_AREA)
    B_1 = cv2.resize(B, dim_1, interpolation=cv2.INTER_AREA)
    plt.imsave(filename + "_1_origin.jpg", np.dstack((R_1, G_1, B_1)))
    G_1 = np.roll(G_1, best_G_0[0] * 4, axis=0)
    G_1 = np.roll(G_1, best_G_0[1] * 4, axis=1)
    B_1 = np.roll(B_1, best_B_0[0] * 4, axis=0)
    B_1 = np.roll(B_1, best_B_0[1] * 4, axis=1)
    img = align_and_combine(R_1, G_1, B_1, normalized_cross_correlation)
    plt.imsave(filename + "_1.jpg", img)
    best_G_1 = best_offset(R_1, G_1, normalized_cross_correlation)
    best_B_1 = best_offset(R_1, B_1, normalized_cross_correlation)
    # level-2
    dim_2 = (col, row)
    R_2 = cv2.resize(R, dim_2, interpolation=cv2.INTER_AREA)
    G_2 = cv2.resize(G, dim_2, interpolation=cv2.INTER_AREA)
    B_2 = cv2.resize(B, dim_2, interpolation=cv2.INTER_AREA)
    plt.imsave(filename + "_2_origin.jpg", np.dstack((R_2, G_2, B_2)))
    G_2 = np.roll(G_2, best_G_0[0] * 16 + best_G_1[0] * 4, axis=0)
    G_2 = np.roll(G_2, best_G_0[1] * 16 + best_G_1[1] * 4, axis=1)
    B_2 = np.roll(B_2, best_B_0[0] * 16 + best_B_1[0] * 4, axis=0)
    B_2 = np.roll(B_2, best_B_0[1] * 16 + best_B_1[1] * 4, axis=1)
    img = align_and_combine(R_2, G_2, B_2, normalized_cross_correlation)
    plt.imsave(filename + "_2.jpg", img)


def part2():
    # TODO: Solution for Q2
    # Task 1: Generate a colour image by splitting 
    # the triptych image and save it 
    img = plt.imread("prokudin-gorskii/00153v.jpg")
    R, G, B = split_triptych(img)
    R = R[0:-1,:]
    img_new = np.dstack((R, G, B))
    plt.imsave("colored_image.jpg", img_new)

    # Task 2: Remove misalignment in the colour channels 
    # by calculating best offset
    img_aligned = align_and_combine(R, G, B, normalized_cross_correlation)
    plt.imsave("aligned_colored.jpg", img_aligned)
    
    # Task 3: Pyramid alignment
    #seoul_tableau
    triptych = "seoul_tableau"
    pyramid_align(triptych)

def RGBtoLAB(image):
    
    image = (image*255).astype(np.uint8)
    imageLAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return imageLAB

def part3():
    # TODO: Solution for Q3
    image_indoor_normal = plt.imread('rubik/indoor.png')
    image_outdoor_normal = plt.imread('rubik/outdoor.png')
    plt.imshow(image_indoor_normal[:,:,0], cmap='gray')
    plt.savefig('image_indoor_normal_R.png')
    plt.imshow(image_indoor_normal[:,:,1], cmap='gray')
    plt.savefig('image_indoor_normal_G.png')
    plt.imshow(image_indoor_normal[:,:,2], cmap='gray')
    plt.savefig('image_indoor_normal_B.png')

    plt.imshow(image_outdoor_normal[:,:,0], cmap='gray')
    plt.savefig('image_outdoor_normal_R.png')
    plt.imshow(image_outdoor_normal[:,:,1], cmap='gray')
    plt.savefig('image_outdoor_normal_G.png')
    plt.imshow(image_outdoor_normal[:,:,2], cmap='gray')
    plt.savefig('image_outdoor_normal_B.png')

    plt.imshow(RGBtoLAB(image_indoor_normal)[:,:,0], cmap='gray')
    plt.savefig('image_indoor_Lab_L.png')
    plt.imshow(RGBtoLAB(image_indoor_normal)[:,:,1], cmap='gray')
    plt.savefig('image_indoor_Lab_aa.png')
    plt.imshow(RGBtoLAB(image_indoor_normal)[:,:,2], cmap='gray')
    plt.savefig('image_indoor_Lab_bb.png')

    plt.imshow(RGBtoLAB(image_outdoor_normal)[:,:,0], cmap='gray')
    plt.savefig('image_outdoor_Lab_L.png')
    plt.imshow(RGBtoLAB(image_outdoor_normal)[:,:,1], cmap='gray')
    plt.savefig('image_outdoor_Lab_aa.png')
    plt.imshow(RGBtoLAB(image_outdoor_normal)[:,:,2], cmap='gray')
    plt.savefig('image_outdoor_Lab_bb.png')
    

def main():
    part1()
    part2()
    part3()


if __name__ == "__main__":
    main()
