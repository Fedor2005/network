import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def countors(img):
    edges = cv.Canny(img, 150, 175)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


img = cv.imread('photo_2021-10-22_17-22-15.jpg', 0)
contours, hierarchy = cv.findContours(cv.Canny(img, 150, 175), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.imshow(contours, hierarchy)