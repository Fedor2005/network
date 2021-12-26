import cv2
import numpy as np


def viewImage(image, name_of_window='cv2'):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def thresh_callback(img):
    threshold1 = 1
    threshold2 = 200
    img = cv2.Canny(img, threshold1, threshold2)
    contours, hierarchy = cv2.findContours(img,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (255, 255, 255)
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)

    return drawing


def thickening_line(img, rad=2):
    drawing = np.full(shape=img.shape, dtype=np.uint8, fill_value=255)
    points = np.argwhere(img == 255)
    for i in points:
        cv2.circle(drawing, i[:2][::-1], rad, (0, 0, 0))
    return drawing


def changing_size(img):
    dsize = (28, 28)
    output = cv2.resize(img, dsize)
    return output


def square(img):
    rsz_img = cv2.resize(img, None, fx=0.25, fy=0.25)  # resize since image is huge
    gray = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    points = np.argwhere(thresh_gray == 0)  # find where the black pixels are
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points)  # create a rectangle around those points
    if w < h:
        w = h
    elif w > h:
        h = w
    x, y, w, h = x - 10, y - 10, w + 20, h + 20  # make the box a little bigger
    crop = gray[y:y + h, x:x + w]  # create a cropped region of the gray image
    retval, thresh_crop = cv2.threshold(crop, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
    return thresh_crop


def main(file_path):
    image = cv2.imread(file_path)
    # viewImage(image)
    image = thresh_callback(image)

    # viewImage(image)
    image = thickening_line(image, rad=16)

    # viewImage(image)
    image = square(image)

    # viewImage(image)
    image = changing_size(image)
    cv2.imwrite('archive/test/test/new.png', image)
    # viewImage(image)

if __name__ == '__main__':
    main('/home/fedor/Desktop/network/test_image/IMG_20220316_215751_658.jpg')