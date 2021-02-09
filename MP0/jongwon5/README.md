design

class ImgUnshredder(src, dst):

"""
member array_dict: dictionary of strips (keys: filename, values: array of image)
member sort_order: deque of sort order from left to right
member pixel_offset: deque of pixel offset from left to right
"""

def combine_strips(self):
"""
param array_dict: dictionary of strips (keys: filename, values: array of image)
param sort_order: deque of sort order from left to right
return combined_img: combined strips
"""

def sort_strips(self, mode="ssd"):
"""
param array_dict
param mode: mode of comparison (either ssd or zncc)
return sort_order
"""

def _ssd(self, img1, img2):
"""
param img1: first array, whose right side will be investigated
param img2: second array, whose left side will be investigated
return ssd: sum of squared L2 distance for all pairs of pixels
"""

def _zncc(self, img1, img2):
"""
param img1: first array, whose right side will be investigated
param img2: second array, whose left side will be investigated
return ssd: AVERAGE zero mean normalized cross correlation for all pairs of pixels (CAUTION: only the average values for non-zero pixels should be applied)
"""

