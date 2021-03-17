import os
import cv2
import numpy as np
import scipy
from scipy import ndimage, signal


def compute_corners(I, window_size=5, ws_nonmax=7, alpha=None):
  """
    Harris corner detector which returns corner response map and non-max suppressed corners.
  Input:
    I : input image, H w W w 3 BGR image
    window_size : window size for operation
    ws_nonmax : window size for nonmax suppression
  Output:
    response : H w W response map in uint8 format
    corners : H w W map in uint8 format _after_ non-max suppression. Each
    pixel stores the score for being a corner. Non-max suppressed pixels
    should have a low / zero-score.
  """

  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
  H = I.shape[0]
  W = I.shape[1]

  # Declare variables to be returned
  response = np.zeros(I.shape)
  corners  = np.zeros(I.shape)
  
  # Create image gradient
  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary='symm')
  Ixx = dx**2
  Ixy = dx*dy
  Iyy = dy**2

  # Enhance gradients
  G = np.sqrt(Ixx + Iyy)
  mask = np.where(G < G.mean(), 0., 1.)

  Ixx = mask * Ixx
  Iyy = mask * Iyy
  
  # Generate guassian filter with the size of `window_size`, whose standard deviation is one third of `window_size // 2`
  w_gaussian = create_gaussian_kernel(window_size, (window_size // 2) / 3)
  
  # Find array of patchs over all pixels
  offset = window_size // 2
  M_dets = np.full(I.shape, np.nan)
  M_traces = np.full(I.shape, np.nan)
  for h in range(offset, H-offset):
    for w in range(offset, W-offset):
      window_Ixx = Ixx[h-offset:h+offset+1, w-offset:w+offset+1]
      window_Ixy = Ixy[h-offset:h+offset+1, w-offset:w+offset+1]
      window_Iyy = Iyy[h-offset:h+offset+1, w-offset:w+offset+1]
      
      window_Ixx = w_gaussian * window_Ixx
      window_Ixy = w_gaussian * window_Ixy
      window_Iyy = w_gaussian * window_Iyy
      
      sum_xx = window_Ixx.sum()
      sum_xy = window_Ixy.sum()
      sum_yy = window_Iyy.sum()

      patch = np.array([[sum_xx, sum_xy],
                        [sum_xy, sum_yy]])
      M_dets[h, w] = np.linalg.det(patch)
      M_traces[h, w] = np.trace(patch)
  
  # Compute mean and stdv over all data
  mu_det = np.nanmean(M_dets)
  std_det = np.nanstd(M_dets)
  mu_tr = np.nanmean(M_traces)
  std_tr = np.nanstd(M_traces)

  # Construct response with zero-mean normalization
  for h in range(offset, H-offset):
    for w in range(offset, W-offset):
      response[h,w] = (M_dets[h,w] - mu_det) / std_det - (M_traces[h,w] - mu_tr) / std_tr
  
  # Clip the response to avoid edge or flat region
  _, response = cv2.threshold(response, 0.001*response.max(), np.inf, cv2.THRESH_TOZERO)
  response = (response - response.min()) / (response.max() - response.min()) * 255.
  response = response.astype(np.uint8)

  # Do non-max suppression with a window size of `ws_nonmax`
  offset = ws_nonmax // 2
  for h in range(offset, H-offset):
    for w in range(offset, W-offset):
      window = response[h-offset:h+offset+1, w-offset:w+offset+1]
      corners[h,w] = response[h,w] if window.max() == response[h,w] else 0

  return response, corners


def create_gaussian_kernel(window_size, sigma):
  """
  Create 2D gaussian kernel with the size of `window_size` and standard deviation of `sigma`
  Input:
    window_size : kernel size to be created
    sigma : standard deviation of gaussian kernel
  Output:
    kernel : a 2D gaussian kernel which has a standard deviation of `sigma` and the squared size of `window_size`
  """
  ax = np.linspace(-(window_size-1)/2, (window_size-1)/2, window_size)
  xx, yy = np.meshgrid(ax, ax)
  kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
  return kernel


if __name__ == "__main__":
  np.set_printoptions(precision=2)

  show_mine = True
  img_names = ['draw_cube_17', '37073', '5904776']
  
  for img_name in img_names:
    img = cv2.imread(os.path.join('data', 'vis', img_name+'.png'))
    
    if show_mine:
      # Case 1. show my result
      _, corners = compute_corners(img)
    else: 
      # Case 2. show in-built harris corner detector
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    
    corners = cv2.dilate(corners, None)
    img[corners>0.01*corners.max()]=[0,0,255]
    cv2.imwrite('harris_'+img_name+'.png', img)