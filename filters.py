import cv2
import numpy as np

"""
Adding filter functions.

OpenCV provides many edge-finding filters, including Laplcaian, Sobel, and Scharr. These filters are supmpposed to turn non-edge regions into black nd turn 
edge regions into white or saturated colors. However, they are prone to misindentifiying noise as edges. This flaw can be mitigated by blurring an image before 
trying to find its edges.  This flaw can be mitigated by blurring an image before trying to find its edges. OpenCV also provides many blurring filters, including
blur (a simple average), medianBLurn and GaussianBlue. The arguments for the edge-finding and blurring filters vary but always include ksize, an odd whle number 
that represents the width and height (in pixels) of a filter's kernel.

For blurring, we will use medianBlur, which is effective in removing digital video noise especially color images. For edge-finding, we will use Laplacian, which
produces bold edge lines, especially in grayscale images. After applying medianBlur, but beofe applying Laplacian, we should convert the image from BGR to grayscale.

Once we have the result of Laplacian, we can invert it to get black edges on a white background. Then, we can normalize it (so that it's values range from 0 to 1)
and then multiply it with the source image to darken the edges.
"""


# A kernel is a set of weights that determines how each output pixel is calculated from a neighborhood of input pixels. Another term for a kernel is a 'convolution matrix'.
# It mixes up or convolves the pixels in a region. Similarly, a kernel-based filter may be called a convolution filter.

# In this function we created blurKsize argument to be used as ksize for medianBlur, while edgeKsize is used as ksize for Lapalacian. With a typical webcam, a
# blurKsize value of 7 and edgeKsize value of 5 might produce the most pleasing effect. But the medianBlur is expensive with a large ksize argument such as 7.

def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


# We add now  two classes 'VConvolutionFilter, will represent a convolution filter in general. A subclass, SharpenFilter, will represent our sharpening flter specifically.

class VConvolutionFilter(object):  # Applies a convolution to V (or all of BGR)
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):  # Apply the filter with a BGR or gray source/destination
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):  # Sharpen filter with a 1-pixel radius
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):  # An edge finding flter with a 1 pixel radius
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):  # A blur filter with a 2-pixel radius
    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):  # Emboss filter with a 1-pixel radius
    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)


class testFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])
        VConvolutionFilter.__init__(self, kernel)


class edgeFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[-1, 1, -1],
                           [-1, 1, -1],
                           [-1, 1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class exoFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[1, 0, -1],
                           [0, 0, 0],
                           [-1, 0, 1]])
        VConvolutionFilter.__init__(self, kernel)