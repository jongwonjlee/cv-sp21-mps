import numpy as np
from scipy import signal, ndimage
import cv2
from math import floor, ceil

def compute_edges_dxdy(I):
  """Returns the norm of dx and dy as the edge response function."""
  I = cv2.GaussianBlur(I, (3,3), 0)
  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
  I = I.astype(np.float32)/255.
  
  dx = ndimage.gaussian_filter1d(I,   sigma=4., order=1)
  dy = ndimage.gaussian_filter1d(I.T, sigma=4., order=1).T
  mag = np.sqrt(dx**2 + dy**2)
  direction = np.arctan2(dy, dx)
  mag = non_max_suppression(mag, direction)

  _, mag = cv2.threshold(mag, 0, np.inf, cv2.THRESH_TOZERO)
  mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag)) * 255.
  mag = mag.astype(np.uint8)

  kernel = np.ones((1,1), np.uint8)
  mag = cv2.morphologyEx(mag, cv2.MORPH_OPEN, kernel)

  return mag


def non_max_suppression(mag, D):
  H, W = mag.shape
  Z = np.zeros((H,W), dtype=np.float32)

  for h in range(1,H-1):
    for w in range(1,W-1):
        angle = D[h,w]
        if 1/4 * np.pi < abs(angle) < 3/4 * np.pi: # project to horizontal axis
          # case 1
          k = 1 / np.tan(angle)
          x = w + k
          y = h + 1 if angle > 0 else h - 1
          p = (ceil(x) - x) * mag[y, floor(x)] + (x - floor(x)) * mag[y, ceil(x)]
          # case 2
          k = - 1 / np.tan(angle)
          x = w - k
          y = h - 1 if angle > 0 else h + 1
          r = (ceil(x) - x) * mag[y, floor(x)] + (x - floor(x)) * mag[y, ceil(x)]
        else: # project to vertical axis
          # case 1
          k = np.tan(angle)
          x = w + 1 if angle > 0 else w - 1
          y = h + k
          p = (ceil(y) - y) * mag[floor(y), x] + (y - floor(y)) * mag[ceil(y), x]
          # case 2
          k = - np.tan(angle)
          x = w - 1 if angle > 0 else w + 1
          y = h - k
          r = (ceil(y) - y) * mag[floor(y), x] + (y - floor(y)) * mag[ceil(y), x]
        
        if max(mag[h,w], p, r) == mag[h,w]:
          Z[h,w] = mag[h,w]
        else:
          Z[h,w] = 0
        
  return Z

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
  
  highThreshold = img.max() * highThresholdRatio
  lowThreshold = highThreshold * lowThresholdRatio
  
  M, N = img.shape
  res = np.zeros((M,N), dtype=np.int32)
  
  weak = np.int32(25)
  strong = np.int32(255)
  
  strong_i, strong_j = np.where(img >= highThreshold)
  zeros_i, zeros_j = np.where(img < lowThreshold)
  
  weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
  
  res[strong_i, strong_j] = strong
  res[weak_i, weak_j] = weak
  
  return (res, weak, strong)

def hysteresis(img, weak, strong=255):
  M, N = img.shape  
  for i in range(1, M-1):
    for j in range(1, N-1):
      if (img[i,j] == weak):
        try:
          if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
              or (img[i, j-1] == strong) or (img[i, j+1] == strong)
              or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
            img[i, j] = strong
          else:
            img[i, j] = 0
        except IndexError as e:
          pass
  return img

if __name__ == "__main__":
    filename = '227092'

    I = cv2.imread(f'{filename}.jpg')
    edge = compute_edges_dxdy(I)
    cv2.imwrite(f'{filename}-implemented.png', edge)