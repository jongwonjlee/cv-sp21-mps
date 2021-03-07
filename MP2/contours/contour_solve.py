import numpy as np
from scipy import signal, ndimage
import cv2
from math import floor

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


def non_max_suppression_original(img, D):
  M, N = img.shape
  Z = np.zeros((M,N), dtype=np.int32)
  angle = D * 180. / np.pi
  angle[angle < 0] += 180

  for i in range(1,M-1):
    for j in range(1,N-1):
      q = 255
      r = 255
      
      #angle 0
      if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
        q = img[i, j+1]
        r = img[i, j-1]
      #angle 45
      elif (22.5 <= angle[i,j] < 67.5):
        q = img[i+1, j-1]
        r = img[i-1, j+1]
      #angle 90
      elif (67.5 <= angle[i,j] < 112.5):
        q = img[i+1, j]
        r = img[i-1, j]
      #angle 135
      elif (112.5 <= angle[i,j] < 157.5):
        q = img[i-1, j-1]
        r = img[i+1, j+1]

      if (img[i,j] >= q) and (img[i,j] >= r):
        Z[i,j] = img[i,j]
      else:
        Z[i,j] = 0

  return Z


def non_max_suppression(mag, D):
  H, W = mag.shape
  Z = np.zeros((H,W), dtype=np.int32)

  for h in range(1,H-1):
    for w in range(1,W-1):
        angle = D[h,w]
        if 1/4 * np.pi < abs(angle) < 3/4 * np.pi: # project to horizontal axis
          # case 1: positive direction of gradient
          k = 1 / np.tan(angle)
          p = (floor(k)-k+1) * mag[h+floor(k),w+floor(k)] + (k-floor(k)) * mag[h+floor(k),w+floor(k)+1]
          # case 2: negative direction of gradient
          k = - 1 / np.tan(angle)
          r = (floor(k)-k+1) * mag[h+floor(k),w+floor(k)] + (k-floor(k)) * mag[h+floor(k),w+floor(k)+1]
        else: # project to vertical axis
          # case 1: positive direction of gradient
          k = np.tan(angle)
          p = (floor(k)-k+1) * mag[h+floor(k),w+floor(k)] + (k-floor(k)) * mag[h+floor(k)+1,w+floor(k)]
          # case 2: negative direction of gradient
          k = - np.tan(angle)
          r = (floor(k)-k+1) * mag[h+floor(k),w+floor(k)] + (k-floor(k)) * mag[h+floor(k)+1,w+floor(k)]
        

        if max(mag[h,w], p, r) == mag[h,w]:
          Z[h,w] = mag[h,w]
        else:
          Z[h,w] = 0
        
  return Z