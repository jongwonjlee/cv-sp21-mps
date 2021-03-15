import numpy as np
import scipy
from scipy import signal
import cv2

def blend(im1, im2, mask):
  """
  description : blend images by pyramid blending
  param im1 : input image 1
  param im2 : input image 2
  param mask : mask for image blending (ones for im1, whereas zeros for im2)
  output out : blended image output
  """

  mask = mask / 255.
  
  gp_im1 = construct_gp(im1)
  gp_im2 = construct_gp(im2)
  gp_mask = construct_gp(mask)

  lp_im1 = construct_lp(gp_im1)
  lp_im2 = construct_lp(gp_im2)
  
  out = combine_lp(gp_mask, lp_im1, lp_im2)
  
  return out

def combine_lp(gp_mask, lp_im1, lp_im2, octave=5):
  """
  description : add up all images in laplacian pyramids with mask
  param gp_mask : gaussian pyramid of mask
  param lp_im1 : laplacian pyramid of input image 1
  param lp_im2 : laplacian pyramid of input image 2
  param octave : number of images comprising pyramids
  output out : blended image output
  """

  assert octave+1 == len(gp_mask) == len(lp_im1) == len(lp_im2)

  lp_blend = [gp_mask[-1]]
  for i in range(octave+1):
    lp_blend.append(gp_mask[octave-i] * lp_im1[i] + (1. - gp_mask[octave-i]) * lp_im2[i])
  
  out = lp_blend[0]
  for i in range(octave+1):
    out = cv2.resize(out, (lp_blend[i+1].shape[1], lp_blend[i+1].shape[0]))
    out = cv2.add(out, lp_blend[i+1])

  return out

## normal order
def construct_gp(im, sig=1., octave=5):
  """
  description : construct a gaussian pyramid with corresponding smoothness and octaves
  param im : input image
  param sig : standard deviation of gaussian filter to be used
  param octave : number of images to comprise the pyramid
  output gp : list of gaussian pyramid, a set of smoothed images, with smoothing and downsampling.
              (note: first element is original image, and others are sequence of downsamples)
  """
  G = im.copy()
  gp = [G]
  
  for i in range(octave):
    G = scipy.ndimage.gaussian_filter(gp[i], sig, order=0, output=None, mode='reflect')
    G = cv2.resize(G, (G.shape[1]//2, G.shape[0]//2))
    gp.append(G)

  return gp

## inverse order
def construct_lp(gp, octave=5):
  """
  description : construct a laplacian pyramid from corresponding gaussian pyramid
  param gp : a gaussian pyramid in concern
  param octave : number of images comprising the pyramid
  output lp : list of laplacian pyramid, a set of difference of gaussian (DoG)
              (note: downsamples in lp is in the order of smallest to largest, which is inverse to gp)
  """

  assert len(gp) == octave+1

  lp = [gp[-1]]
  
  for i in range(octave, 0, -1):
    G_upsampled = cv2.resize(gp[i], (gp[i-1].shape[1], gp[i-1].shape[0]))
    L = cv2.subtract(gp[i-1], G_upsampled)
    lp.append(L)
  
  return lp