import numpy as np

"""
To identify outliers in the disparity map, we first find the median using np.median which takes an array as an argument. If the array is of an odd length,
median returns the value that would lie i nthe middle of the array if the array were sorted. If the arrray is of an even length, median returns the average 
of the two valeurs that would be sorted nearest to the middle of the array.
"""

def createMedianMask(disparityMap, validDepthMask, rect = None):
    """Return a mask selecting the median layer, plus shadows."""
    if rect is not None:
        x, y, w, h = rect
        disparityMap = disparityMap[y:y+h, x:x+w]
        validDepthMask = validDepthMask[y:y+h, x:x+w]
    median = np.median(disparityMap)
    return np.where((validDepthMask == 0) | \
                       (abs(disparityMap - median) < 12), 255, 0).astype(np.uint8)

"""
To generate a mask based on per-pixel BOolean operations, we use np.where width three arguments. In the fist arguemetn , where takes an array whose elements are 
evaluated for thruth and falsity. An output array of the same dimensions is returned. Wherever an element in the input array is True, the where functions's second 
argument is assigned to to the corrrespondig element in the output array. Conversely, wherever an element in the input aray os False, the where function's third 
argument is assigned to the corresponding element in the outpyt array.  
"""

