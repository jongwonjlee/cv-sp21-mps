import numpy as np
import scipy
from scipy import signal
import cv2

def blend(im1, im2, mask):
  mask = mask / 255.
  
  gp_im1 = construct_gp(im1)
  gp_im2 = construct_gp(im2)
  gp_mask = construct_gp(mask)

  lp_im1 = construct_lp(gp_im1)
  lp_im2 = construct_lp(gp_im2)
  
  out = combine_lp(gp_mask, lp_im1, lp_im2)
  
  return out

def combine_lp(gp_mask, lp_im1, lp_im2, octave=5):
  assert octave+1 == len(gp_mask) == len(lp_im1) == len(lp_im2)

  lp_blend = [gp_mask[-1]]
  for i in range(0, octave+1):
    lp_blend.append(gp_mask[octave-i] * lp_im1[i] + (1. - gp_mask[octave-i]) * lp_im2[i])
  
  im_result = lp_blend[0]
  for i in range(0, octave):
    im_result = cv2.resize(im_result, (lp_blend[i+1].shape[1], lp_blend[i+1].shape[0]))
    im_result = cv2.add(im_result, lp_blend[i+1])

  return im_result

## normal order
def construct_gp(im, sig=1., octave=5):
  G = im.copy()
  gp = [G]
  
  for i in range(octave):
    G = scipy.ndimage.gaussian_filter(gp[i], sig, order=0, output=None, mode='reflect')
    G = cv2.resize(G, (G.shape[1]//2, G.shape[0]//2))
    gp.append(G)

  return gp

## inverse order
def construct_lp(gp, octave=5):
  assert len(gp) == octave+1

  lp = [gp[-1]]
  
  for i in range(octave, 0, -1):
    G_upsampled = cv2.resize(gp[i], (gp[i-1].shape[1], gp[i-1].shape[0]))
    L = cv2.subtract(gp[i-1], G_upsampled)
    lp.append(L)
  
  return lp