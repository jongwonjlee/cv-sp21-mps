import os, time
import cv2
import numpy as np

import argparse
from collections import deque

class ImgUnshredder:
    def __init__(self, src_dir):
        self.strips_unsorted = dict()   # dictionary of unsorted strips (keys: filename, values: array of strip data)
        self.strips_sorted = deque()    # deque of sorted strips from left to right
        self.relative_offset = deque()  # deque of relative pixel offset from left to right
        
        # size of canvas to save combined strips
        self.H = None
        self.W = None
        self.C = None

        # read strips in `src_dir` and prepare to sorting
        self._read_strips(src_dir)
        self.strips_sorted.append(self.strips_unsorted.popitem()[-1])

    def combine_strips(self):
        """
        description : combine images in the order of self.strips_sorted and relative offset of self.relative_offset
        return combined_img : combined strips
        """
        # create a canvas to save the result
        combined_img = np.zeros([self.H, self.W, self.C])
        
        # x: current horizontal position at which combined_img is painted
        # y: current vertical position at which combined_img is painted (calculated by accumulating relative offset over different strips)
        x = 0
        y = 0
        for i, img in enumerate(self.strips_sorted):
            try:
                offset = self.relative_offset[i]
            except:
                offset = 0
            h, w, _ = np.shape(img)
            combined_img[max(y,0):min(h+y,self.H), x:x+w, :] = img[max(-y,0): min(self.H-y,h), :, :]
            x += w
            y += offset
        return combined_img
        
    def sort_strips_random(self):
        """
        description : sort strips randomly
        """
        while not len(self.strips_unsorted) == 0:
            self.strips_sorted.append(self.strips_unsorted.popitem()[1])

    def sort_strips_ssd(self):
        """
        description : sort strips using (negative of) the sum of squared differences
        """
        # do until all strips are sorted
        while not len(self.strips_unsorted) == 0:
            similarest_filename = None
            similarest_score = -np.inf
            similarest_direction = None

            # compute SSD for every unsorted strip w.r.t. the set of sorted strips
            for filename, img in self.strips_unsorted.items():
                # (1) check SSD between the rightmost side of sorted strips and a randomly selected strip from the unsorted set
                arr1 = np.float32(self.strips_sorted[-1][:, -1, :])
                arr2 = np.float32(img[:, 0, :])
                nssd = self._nssd(arr1, arr2)
                if similarest_score < nssd:
                    similarest_filename = filename
                    similarest_score = nssd
                    similarest_direction = "right"
                # (2) check SSD between the randomly selected strip from the unsorted set and the leftmost side of sorted strips
                arr1 = np.float32(img[:, -1, :])
                arr2 = np.float32(self.strips_sorted[0][:, 0, :])
                nssd = self._nssd(arr1, arr2)
                if similarest_score < nssd:
                    similarest_filename = filename
                    similarest_score = nssd
                    similarest_direction = "left"
            
            # move the selected strip from the unsorted set to the sorted set
            if similarest_direction == "left":
                self.strips_sorted.appendleft(self.strips_unsorted.pop(similarest_filename))
            else:
                self.strips_sorted.append(self.strips_unsorted.pop(similarest_filename))

    def sort_strips_zncc(self):
        """
        description : sort strips using zero-mean normalized cross correlation
        """
        # do until all strips are sorted
        while not len(self.strips_unsorted) == 0:
            similarest_filename = None
            similarest_score = -np.inf
            similarest_direction = None
            similarest_offset = None

            # compute ZNCC for every unsorted strip w.r.t. the set of sorted strips
            for filename, img in self.strips_unsorted.items():
                # (1) ZNCC between the rightmost side of sorted strips and a randomly selected strip from the unsorted set
                strip1 = np.expand_dims(self.strips_sorted[-1][:, -1, :], axis=1)
                strip2 = np.expand_dims(img[:, 0, :], axis=1)
                strip1 = cv2.cvtColor(strip1, cv2.COLOR_BGR2GRAY)
                strip2 = cv2.cvtColor(strip2, cv2.COLOR_BGR2GRAY)
                strip1 = strip1.astype(np.float32)
                strip2 = strip2.astype(np.float32)

                # (1-1) compare all ZNCC with offset ranging from -offset_max to offset_max
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
                # (2) ZNCC between the randomly selected strip from the unsorted set and the leftmost side of sorted strips
                strip1 = np.expand_dims(img[:, -1, :], axis=1)
                strip2 = np.expand_dims(self.strips_sorted[0][:, 0, :], axis=1)
                strip1 = cv2.cvtColor(strip1, cv2.COLOR_BGR2GRAY)
                strip2 = cv2.cvtColor(strip2, cv2.COLOR_BGR2GRAY)
                strip1 = strip1.astype(np.float32)
                strip2 = strip2.astype(np.float32)

                # (2-1) compare all ZNCC with offset ranging from -offset_max to offset_max
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
            
            # move the selected strip from the unsorted set to the sorted set and record the relative offset between the left to the right
            if similarest_direction == "left":
                self.strips_sorted.appendleft(self.strips_unsorted.pop(similarest_filename))
                self.relative_offset.appendleft(similarest_offset)
            else:
                self.strips_sorted.append(self.strips_unsorted.pop(similarest_filename))
                self.relative_offset.append(similarest_offset)
    
    def _read_strips(self, src_dir):
        """
        description : read strips in `src_dir` and save them in self.strips_unsorted, a dictionary with keys of filename and values of image data
        param src_dir : source directory from which strips are read
        """
        assert os.path.exists(src_dir)
        
        # read each strip to decide the canvas size and add filename and strip data to self.strips_unsorted
        self.W = 0
        self.H = 0
        for filename in os.listdir(src_dir):
            img = cv2.imread(os.path.join(src_dir, filename))
            self.strips_unsorted[filename] = img
            h, w, self.C = np.shape(img)
            self.W = self.W + w
            self.H = max(self.H, h)
        
    def _nssd(self, arr1, arr2):
        """
        param arr1 : first array to be investigated (shape: (?, 3))
        param arr2 : second array to be investigated (shape: (?, 3))
        return nssd : NEGATIVE sum of squared L2 distance for all pairs of pixels
        """
        assert arr1.shape == arr2.shape
        assert np.shape(arr1)[-1] == np.shape(arr2)[-1] == 3
        assert arr1.dtype == np.dtype(np.float32) and arr2.dtype == np.dtype(np.float32)
        
        return -np.sum(np.linalg.norm(arr1 - arr2, axis=1))

    def _zncc(self, arr1, arr2):
        """
        param arr1 : first array to be investigated (shape: (?,))
        param arr2 : second array to be investigated (shape: (?,))
        return avg_zncc : AVERAGE zero mean normalized cross correlation for all pairs of pixels
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
    parser.add_argument('--src-dir', type=str, help='directory to read shredded images', default="../shredded-images/")
    parser.add_argument('--dst-dir', type=str, help='directory to write concatenated image', default="results/")
    parser.add_argument('--img-name', type=str, help='image to concatenate', \
        choices=['simple_almastatue', 'simple_larry-roberts', 'hard_almastatue', 'hard_texture', 'hard_text', 'hard_building'])
    parser.add_argument('--mode', type=str, help='matching method to use', choices=[None, 'ssd', 'zncc'], default=None)

    args = parser.parse_args()
    print(args)

    # create a ImgUnshredder object
    src_dir = os.path.join(args.src_dir, args.img_name)
    unshredder = ImgUnshredder(src_dir)

    # do corresponding sorting method: None, ssd, or zncc
    tic = time.time()
    if args.mode == None:
        unshredder.sort_strips_random()
        args.img_name += "_random"
    elif args.mode == "ssd":
        unshredder.sort_strips_ssd()
    elif args.mode == "zncc":
        unshredder.sort_strips_zncc()
    print(f"collapsed time: {time.time() - tic:.4f} [s]")

    # combine sorted strips
    img = unshredder.combine_strips()

    # save the result
    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)
    cv2.imwrite(os.path.join(args.dst_dir, args.img_name + ".png"), img)
    