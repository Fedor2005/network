# image = cv2.imread('photo_2021-10-22_17-22-15.jpg')
#     print(image.shape)
#     image = thresh(image)
#     cv2.imwrite('img1.png', image)
#     viewImage(image)
#     image = thickening_line(image, rad=8)
#     image = resize(image)
#     cv2.imwrite('img.png', image)
#     viewImage(image)
import cv2
import numpy as np
import random as rnd
import os
from prdictofobrabotannimg import predict


def viewImage(image, name_of_window='cv2'):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def thresh(image, val1=100, val2=255, val3=0):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray_image, val1, val2, val3)
    return threshold_image


def find_contours(img):
    # blurred = cv2.GaussianBlur(image, (3, 3), 0)
    # T, thresh_img = cv2.threshold(blurred, 215, 255,
    #                               cv2.THRESH_BINARY)
    cnts, h = cv2.findContours(img,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def thresh_callback(img):
    threshold1 = 100
    threshold2 = 200
    img = cv2.Canny(img, threshold1, threshold2)
    # Find contours
    contours, hierarchy = cv2.findContours(img,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rnd.randint(0, 256), rnd.randint(0, 256), rnd.randint(0, 256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    viewImage(drawing)
    return drawing


def thickening_line(img, rad=3):
    drawing = np.full(shape=img.shape, dtype=np.uint8, fill_value=255)
    k_row = 0
    for row in img:
        k_column = 0
        for column in row:
            if column == 0:
                cv2.circle(drawing, (k_row, k_column), rad, (0, 0, 0), rad)
            k_column += 1
        k_row += 1
    return drawing


def changing_size(img):
    dsize = (28, 28)
    output = cv2.resize(img, dsize)
    return output


def contours(img):
    edges = cv2.Canny(img, 150, 175)
    return edges


def resize(img):
    return img[110:960, 100:950]#низ 950 вверх100 право960 лево 200


if __name__ == '__main__':
    image = cv2.imread('img.png')
    image = cv2.resize(image, [28,28])
    cv2.imwrite('img1.png', image)