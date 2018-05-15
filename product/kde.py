import numpy as np
#import matplotlib.mlab as mlab
from matplotlib import pyplot as plt
import sys
import seaborn
from paper import read_geotiff
#from scipy.stats import norm

np.set_printoptions(threshold=np.nan)

def get_bin_argmax(bin):
    return np.argmax(bin)

def get_bins(data, x, y):
    return np.transpose(data)[y,x]

if __name__ == '__main__':
    filename = sys.argv[1]
    data = np.array(read_geotiff(filename))
    
    print(data.shape) 
        
    _x = data.shape[1]
    _y = data.shape[2]

    bin_grid = np.matrix([[0]*_x]*_y)
    bin_list = []
    for i in range(_x):
        for j in range(_y):
            bins = get_bins(data, i, j)
            _max = get_bin_argmax(bins) 
            bin_list.append(_max)
            #if _max > 23:
            #    _max = 45 - _max
            bin_grid[j, i] = _max
            
    
    #seaborn.kdeplot(bin_list)
    #plt.show()
    #plt.imshow(bin_grid)
    #plt.show()
