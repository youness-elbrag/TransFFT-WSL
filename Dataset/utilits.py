import numpy as np 
import cv2
from skimage import data
from skimage import filters
from skimage.color import rgb2gray


# https://www.kaggle.com/code/snnclsr/roi-extraction-using-opencv
def crop_coords(img):
    """
    Crop ROI from image.
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)

# numpy
def histogram_equalization(img):
    m = int(np.max(img))
    hist = np.histogram(img, bins=m+1, range=(0, m+1))[0]
    # bước 1: tính pdf
    hist = hist/img.size
    # bước 2: tính cdf
    cdf = np.cumsum(hist)
    # bước 3: lập bảng thay thế
    s_k = (255 * cdf)
    # ảnh mới
    img_new = np.array([s_k[i] for i in img.ravel()]).reshape(img.shape)
    return img_new

def normalization_(img):
    """
    WindowCenter and normalize pixels in the breast ROI.
    return: numpy array of the normalized image
    """
    # Convert to float to avoid overflow or underflow losses.
    image_2d = img.astype(float)
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    # Convert to uint
    normalized = np.uint8(image_2d_scaled)
    return normalized.astype(np.float32)