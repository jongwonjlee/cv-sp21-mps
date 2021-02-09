import sys, os
import numpy as np
import cv2
import argparse
from collections import deque

import random
import time

class ImgUnshredder:
    def __init__(self, src):
        """
        member array_dict: dictionary of strips (keys: filename, values: array of image)
        member sort_order: deque of sort order from left to right
        member relative_offset: deque of pixel offset from left to right
        """
        self.src = src
        self.strips_unsorted = dict()
        self.strips_sorted = deque()
        self.relative_offset = deque()
        
        self._read_strips()

    def combine_strips(self):
        """
        param array_dict: dictionary of strips (keys: filename, values: array of image)
        param sort_order: deque of sort order from left to right
        return combined_img: combined strips
        """
        combined_img = np.zeros([self.H, self.W, self.C])
        
        w = 0
        d = 0
        for i, img in enumerate(self.strips_sorted):
            try:
                offset = self.relative_offset[i]
            except:
                offset = 0
            H, W, _ = np.shape(img)
            combined_img[max(d, 0) : min(H + d, self.H), w : w + W, :] \
                = img[max(-d, 0): min(self.H - d, H), :, :]
            w = w + W
            d = d + offset
        return combined_img
        
    def sort_strips_random(self):
        while not len(self.strips_unsorted) == 0:
            self.strips_sorted.append(self.strips_unsorted.popitem()[1])

    def sort_strips_ssd(self):
        """
        param array_dict
        return sort_order
        """
        while not len(self.strips_unsorted) == 0:
            similarest_filename = None
            similarest_score = -np.inf
            similarest_direction = None
            for filename, img in self.strips_unsorted.items():
                # check right side first
                arr1 = np.float32(self.strips_sorted[-1][:, -1, :])
                arr2 = np.float32(img[:, 0, :])
                ssd = self._ssd(arr1, arr2)
                if similarest_score < ssd:
                    similarest_filename = filename
                    similarest_score = ssd
                    similarest_direction = "right"
                # check left side
                arr1 = np.float32(img[:, -1, :])
                arr2 = np.float32(self.strips_sorted[0][:, 0, :])
                ssd = self._ssd(arr1, arr2)
                if similarest_score < ssd:
                    similarest_filename = filename
                    similarest_score = ssd
                    similarest_direction = "left"
            
            if similarest_direction == "left":
                self.strips_sorted.appendleft(self.strips_unsorted.pop(similarest_filename))
            else:
                self.strips_sorted.append(self.strips_unsorted.pop(similarest_filename))

    def sort_strips_zncc(self):
        """
        param array_dict
        return sort_order
        """
        while not len(self.strips_unsorted) == 0:
            similarest_filename = None
            similarest_score = -np.inf
            similarest_direction = None
            similarest_offset = None
            for filename, img in self.strips_unsorted.items():
                # check right side
                strip1 = np.expand_dims(self.strips_sorted[-1][:, -1, :], axis=1)
                strip2 = np.expand_dims(img[:, 0, :], axis=1)
                strip1 = cv2.cvtColor(strip1, cv2.COLOR_BGR2GRAY)
                strip2 = cv2.cvtColor(strip2, cv2.COLOR_BGR2GRAY)
                strip1 = strip1.astype(np.float32)
                strip2 = strip2.astype(np.float32)
                offset_max = int(strip1.size * 0.2)
                for offset in range(-offset_max, offset_max):
                    arr1 = strip1[max(0, offset): min(strip1.size, strip2.size + offset)].squeeze()
                    arr2 = strip2[max(0, -offset): min(strip1.size - offset, strip2.size)].squeeze()
                    zncc = self._zncc(arr1, arr2)
                    if similarest_score < zncc:
                        similarest_filename = filename
                        similarest_score = zncc
                        similarest_direction = "right"
                        similarest_offset = offset
                # check left side
                strip1 = np.expand_dims(img[:, -1, :], axis=1)
                strip2 = np.expand_dims(self.strips_sorted[0][:, 0, :], axis=1)
                strip1 = cv2.cvtColor(strip1, cv2.COLOR_BGR2GRAY)
                strip2 = cv2.cvtColor(strip2, cv2.COLOR_BGR2GRAY)
                strip1 = strip1.astype(np.float32)
                strip2 = strip2.astype(np.float32)
                offset_max = int(strip1.size * 0.2)
                for offset in range(-offset_max, offset_max):
                    arr1 = strip1[max(0, offset): min(strip1.size, strip2.size + offset)].squeeze()
                    arr2 = strip2[max(0, -offset): min(strip1.size - offset, strip2.size)].squeeze()
                    zncc = self._zncc(arr1, arr2)
                    if similarest_score < zncc:
                        similarest_filename = filename
                        similarest_score = zncc
                        similarest_direction = "left"
                        similarest_offset = offset
            
            if similarest_direction == "left":
                self.strips_sorted.appendleft(self.strips_unsorted.pop(similarest_filename))
                self.relative_offset.appendleft(similarest_offset)
            else:
                self.strips_sorted.append(self.strips_unsorted.pop(similarest_filename))
                self.relative_offset.append(similarest_offset)
    
    def _read_strips(self):
        """
        param src: source directory to read strips
        return array_dict: dictionary of strips (keys: filename, values: array of image)
        """
        try:
            os.path.exists(self.src)
        except IOError:
            print("FILE DOES NOT EXIST IN ", self.src)
        
        self.W = 0
        self.H = 0
        for filename in os.listdir(self.src):
            img = cv2.imread(os.path.join(self.src, filename))
            self.strips_unsorted[filename] = img
            h, w, self.C = np.shape(img)
            self.W = self.W + w
            self.H = max(self.H, h)
        
        self.strips_sorted.append(self.strips_unsorted.popitem()[-1])
            

    def _ssd(self, arr1, arr2):
        """
        param arr1: first array to be investigated
        param arr2: second array to be investigated
        return ssd: NEGATIVE sum of squared L2 distance for all pairs of pixels
        """
        assert arr1.dtype == np.dtype(np.float32) and arr2.dtype == np.dtype(np.float32)
        
        return -np.sum(np.linalg.norm(arr1 - arr2, axis=1))

    def _zncc(self, arr1, arr2):
        """
        param arr1: first array to be investigated
        param arr2: second array to be investigated
        return zncc: AVERAGE zero mean normalized cross correlation for all pairs of pixels (CAUTION: only the average values for non-zero pixels should be applied)
        """
        assert arr1.shape == arr2.shape
        assert arr1.ndim == 1 and arr2.ndim == 1
        assert arr1.dtype == np.dtype(np.float32) and arr2.dtype == np.dtype(np.float32)
        
        avg1 = np.average(arr1)
        avg2 = np.average(arr2)
        std1 = np.std(arr1)
        std2 = np.std(arr2)

        N = arr1.size
        sum_zncc = 0
        for n in range(N):
            sum_zncc += (arr1[n] - avg1) * (arr2[n] - avg2) / (std1 * std2)
        
        return sum_zncc / N
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parameters for image processing')
    parser.add_argument('--src-path', type=str, help='directory to read shredded images', default="MP0/shredded-images/")
    parser.add_argument('--dst-path', type=str, help='directory to write concatenated image', default="MP0/jongwon5/result")
    parser.add_argument('--img-name', type=str, help='image to concatenate', \
        choices=['simple_almastatue', 'simple_larry-roberts', 'hard_almastatue', 'hard_texture', 'hard_text', 'hard_building'])
    parser.add_argument('--mode', type=str, help='mode to use', choices=[None, 'ssd', 'zncc'], default=None)

    args = parser.parse_args()

    # create a ImgUnshredder object
    src = os.path.join(args.src_path, args.img_name)
    unshredder = ImgUnshredder(src)

    # do corresponding sorting method: None, ssd, or zncc
    if args.mode == None:
        unshredder.sort_strips_random()
        args.img_name += "_random"
    elif args.mode == "ssd":
        unshredder.sort_strips_ssd()
    elif args.mode == "zncc":
        unshredder.sort_strips_zncc()
    
    # combine sorted strips
    img = unshredder.combine_strips()

    # save the result
    if not os.path.exists(args.dst_path):
        os.makedirs(args.dst_path)
    cv2.imwrite(os.path.join(args.dst_path, args.img_name + ".png"), img)
    