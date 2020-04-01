import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
from utils import visualization_utils


def get_image_angle(image: np.ndarray, count_crops: int=4, minLineLength=80, is_draw: bool =True):    
    def get_line(image: np.ndarray):              
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 100, 180, apertureSize = 3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=minLineLength, maxLineGap=5)
        if lines is not None:
            length = image.shape[0]
            angles = []
            for x1, y1, x2, y2 in lines[0]:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)
            parts_angles.append(angles)
            median_angle = np.median(angles)
            img_rotated = ndimage.rotate(image, median_angle)
            if is_draw:
                fig = plt.figure(figsize=(12,12)) 
                fig.add_subplot(1,3,1)
                plt.imshow(image)
                fig.add_subplot(1,3,2)
                plt.imshow(edges)
                fig.add_subplot(1,3,3)
                plt.imshow(img_rotated) 
                plt.show()
    parts_angles = []
    for x in range(0, image.shape[0]-1, image.shape[0] // count_crops):
        for y in range(0, image.shape[1]-1, image.shape[1] // count_crops):
            get_line(image[x:x+image.shape[0]//count_crops, y:y+image.shape[1]//count_crops, :])
    return np.median([a if a>0 else 90+a for a in np.array(parts_angles).flatten()])

def rotate_image(image, angle, size):
    h, w = image.shape[:2]
    image_center = (w/2, h/2)

    if size is None:
        radians = math.radians(angle)
        sin = math.sin(radians)
        cos = math.cos(radians)
        size = (int((h * abs(sin)) + (w * abs(cos))), int((h * abs(cos)) + (w * abs(sin))))
        rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)
        rotation_matrix[0, 2] += ((size[0] / 2) - image_center[0])
        rotation_matrix[1, 2] += ((size[1] / 2) - image_center[1])
    else:
        rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)

    return cv2.warpAffine(image, rotation_matrix, size)

image_url = f'{DATASET_DIR}/filename.png'
image = cv2.imread(image_url)   
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (1024, 1024))
image_origin_size = image.shape[:2]
angle = get_image_angle(image, count_crops=4, minLineLength=30, is_draw=False)
if angle  > 0 :
    image = rotate_image(image, angle, None)
else:
    print('Угол не определен для изображения')



