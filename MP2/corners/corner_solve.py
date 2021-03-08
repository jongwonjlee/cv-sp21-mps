import os
import cv2
import numpy as np
import scipy
from scipy import ndimage, signal


def compute_corners(I, img_name='default', window_size=9, ws_nonmax=3, sigma=1, alpha=0.04, thres=10000):
  """
    Harris corner detector which returns corner response map and non-max suppressed corners.
  Input:
    I : input image, H w W w 3 BGR image
    window_size : window size for operation
    ws_nonmax : window size for nonmax suppression
    sigma : standard deviation for 2D gaussian kernel used for weighting
    alpha : hyperparameter for R = det(M) - alpha * tr(M) ** 2
    thres : threshold for R
  Output:
    response: H w W response map in uint8 format
    corners: H w W map in uint8 format _after_ non-max suppression. Each
    pixel stores the score for being a corner. Non-max suppressed pixels
    should have a low / zero-score.
  """
  I = cv2.GaussianBlur(I, (window_size,window_size), 0)
  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
  H = I.shape[0]
  W = I.shape[1]

  response = np.zeros((H,W))
  corners  = np.zeros((H,W))

  dx = ndimage.sobel(I, axis=0)
  dy = ndimage.sobel(I, axis=1)
  Ixx = dx**2
  Ixy = dx*dy
  Iyy = dy**2

  # Generate guassian kernel
  window_gkern = create_gaussian_kernel(window_size, sigma)
  offset = window_size // 2

  # Find corner candidates
  for h in range(offset, H-offset):
    for w in range(offset, W-offset):
      window_Ixx = Ixx[h-offset:h+offset+1, w-offset:w+offset+1]
      window_Ixy = Ixy[h-offset:h+offset+1, w-offset:w+offset+1]
      window_Iyy = Iyy[h-offset:h+offset+1, w-offset:w+offset+1]
      
      window_Ixx = window_gkern * window_Ixx
      window_Ixy = window_gkern * window_Ixy
      window_Iyy = window_gkern * window_Iyy
      
      sum_xx = window_Ixx.sum()
      sum_xy = window_Ixy.sum()
      sum_yy = window_Iyy.sum()

      M = np.array([[sum_xx, sum_xy],
                    [sum_xy, sum_yy]])
      
      R = np.linalg.det(M) - alpha * (np.trace(M) ** 2)
      response[h,w] = R

  # Clip response
  response = np.clip(response, thres, np.inf)
  response = (response - response.min()) / (response.max() - response.min()) * 255.
  response = response.astype(np.uint8)

  # Do non-max suppression
  offset = ws_nonmax // 2
  
  for h in range(offset, H-offset):
    for w in range(offset, W-offset):
      window = response[h-offset:h+offset+1, w-offset:w+offset+1]
      
      if window.max() == response[h,w]:
          corners[h,w] = response[h, w]
      else:
        corners[h,w] = 0
  corners = corners.astype(np.uint8)

  # FIXME: For debug
  # cv2.imwrite('respon_'+img_name+'.png', response)
  # cv2.imwrite('corner_'+img_name+'.png', corners)

  return response, corners

def create_gaussian_kernel(window_size, sig):
  """
  Create 2D gaussian kernel with the size of `window_size` and standard deviation of `sig`
  Input:
    window_size: kernel size to be created
    sig: standard deviation of gaussian kernel
  Output:
    kernel: a 2D gaussian kernel which has a standard deviation of `sig` and the squared size of `window_size`
  """
  ax = np.linspace(-(window_size-1)/2, (window_size-1)/2, window_size)
  xx, yy = np.meshgrid(ax, ax)
  kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
  return kernel

if __name__ == "__main__":
  np.set_printoptions(precision=2)

  show_mine = False
  img_names = ['draw_cube_17', '37073', '5904776']
  
  for img_name in img_names:
    img = cv2.imread(os.path.join('data', 'vis', img_name+'.png'))
    
    if show_mine:
      # Case 1. show my result
      _, corners = compute_corners(img, img_name)
    else: 
      # Case 2. show in-built harris corner detector
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
      corners = cv2.dilate(corners, None)
      # ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
      # dst = np.uint8(dst)
    
    img[corners>0.01*corners.max()]=[0,0,255]
    cv2.imwrite('harris_'+img_name+'.png', img)