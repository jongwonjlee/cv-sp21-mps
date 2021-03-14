import numpy as np
from scipy import signal, ndimage
import cv2
from math import floor, ceil

def compute_edges_dxdy(I):
  """Returns the norm of dx and dy as the edge response function."""
  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
  I = I.astype(np.float32)/255.
  
  ### question 1
  # dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same')
  # dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same')
  
  ### question 2
  # dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
  # dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary='symm')
  
  ### question 3
  dx = ndimage.gaussian_filter1d(I,   sigma=4., order=1)
  dy = ndimage.gaussian_filter1d(I.T, sigma=4., order=1).T
  
  mag = np.sqrt(dx**2 + dy**2)
  mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag))
  mag = mag * 255.
  mag = np.clip(mag, 0, 255)
  mag = mag.astype(np.uint8)

  # question 4
  direction = np.arctan2(dy, dx)
  mag = non_max_suppression(mag, direction)

  return mag


def non_max_suppression_original(mag, D):
  M, N = mag.shape
  Z = np.zeros((M,N), dtype=np.int32)
  angle = D * 180. / np.pi
  angle[angle < 0] += 180

  for h in range(1,M-1):
    for w in range(1,N-1):
      #angle 0
      if (0 <= angle[h,w] < 22.5) or (157.5 <= angle[h,w] <= 180):
        p = mag[h, w+1]
        r = mag[h, w-1]
      #angle 45
      elif (22.5 <= angle[h,w] < 67.5):
        p = mag[h+1, w-1]
        r = mag[h-1, w+1]
      #angle 90
      elif (67.5 <= angle[h,w] < 112.5):
        p = mag[h+1, w]
        r = mag[h-1, w]
      #angle 135
      elif (112.5 <= angle[h,w] < 157.5):
        p = mag[h-1, w-1]
        r = mag[h+1, w+1]
      
      if max(mag[h,w], p, r) == mag[h,w]:
        Z[h,w] = mag[h,w]
      else:
        Z[h,w] = 0

  return Z


def non_max_suppression(mag, D):
  H, W = mag.shape
  Z = np.zeros((H,W), dtype=np.int32)

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

if __name__ == "__main__":
    filename = '227092'

    I = cv2.imread(f'{filename}.jpg')
    edge = compute_edges_dxdy(I)
    cv2.imwrite(f'{filename}-mine.png', edge)