import argparse
import os
import cv2
import numpy as np
import tensorflow as tf


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def resize_img(frame):
    resize = ResizeWithAspectRatio(frame, width=720)
    return np.asarray(resize)


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            # Compute perimeter of contour and perform contour approximation
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def reorder(points):
    points = points.reshape((4, 2))
    pointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)

    pointsNew[0] = points[np.argmin(add)]
    pointsNew[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]

    return pointsNew


def process_img(img_path):
    frame_orig = cv2.imread(img_path)
    # Replace the code below to show only the check and apply transform.
    resized = resize_img(frame_orig)
    height = resized.shape[0]
    width = resized.shape[1]
    frame_result = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Image processing
    gray = cv2.cvtColor(frame_result, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur, 10, 100)
    kernel = np.ones((3, 3))
    imgDial = cv2.dilate(edges, kernel, iterations=2)
    imgErode = cv2.erode(imgDial, kernel, iterations=1)
    # Replace the code above.

    # Find all contours
    imgContours = resized.copy()
    contours, h = cv2.findContours(
        imgErode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    cv2.drawContours(imgContours, hull_list, -1, (0, 0, 255), 5)

    # Find biggest contours
    biggest, maxArea = biggestContour(hull_list)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgContours, biggest, -1, (255, 0, 0), 20)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(resized, matrix, (width, height))
        cv2.imshow("Result", imgWarpColored)
    cv2.imshow("Original", resized)
    # cv2.imshow("filter", imgErode)
    # cv2.imshow("contour", imgContours)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check prepartion project")
    parser.add_argument('--input_folder', type=str,
                        default='samples', help='check images folder')

    args = parser.parse_args()
    input_folder = args.input_folder

    for check_img in os.listdir(input_folder):
        img_path = os.path.join(input_folder, check_img)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            process_img(img_path)
