from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

class Dataloader(keras.utils.Sequence):
    def __init__(self, lrs, hrs, count, grid = int(2**8)):
        self.hrs = iter(hrs)
        self.lrs = iter(lrs)
        self.count = count
        self.grid = grid
    
    def cut_image(self):
        hr_list = []
        lr_list = []
        for idx , paths in enumerate(zip(self.hrs, self.lrs)):
            hr = plt.imread(paths[0])
            lr = plt.imread(paths[1])
            
            block_nums = int(hr.shape[0] / self.grid)

            hr_shape = int(hr.shape[0] / block_nums)
            lr_shape = int(lr.shape[0] / block_nums)
      

            for w in range(block_nums):
                for h in range(block_nums):
                    hr_list.append(hr[w*hr_shape:(w+1)*hr_shape, h*hr_shape:(h+1)*hr_shape,:])
                    lr_list.append(lr[w*lr_shape:(w+1)*lr_shape, h*lr_shape:(h+1)*lr_shape,:])

            if idx == self.count-1:
                return np.array(lr_list), np.array(hr_list)
            
 
            