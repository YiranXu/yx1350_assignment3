# 
# A CUDA version to calculate the Mandelbrot set
#
from numba import cuda
import math
import numpy as np
from pylab import imshow, show
@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the 
    Mandelbrot set given a fixed number of iterations.
    '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@cuda.jit
def compute_mandel(min_x, max_x, min_y, max_y, image, iters):
    '''
    This kernel(each thread) will iterate over a small block of elements
    The size of block each thread needs to fill: 
    width=ceil of (width of image/total number of threads in x dimension)
    height=ceil of (height of image/totl number of threads in y dimension)
    '''
    #get absolute position for this thread
    id_x,id_y=cuda.grid(2)
    #get image size and pixel size
    height = image.shape[0]
    width = image.shape[1]
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    #total number of threads in x dimension and y dimension 
    total_x=cuda.gridDim.x*cuda.blockDim.x
    total_y=cuda.gridDim.y*cuda.blockDim.y
    #size of block each thread needs to fill:
    each_x=int(math.ceil(width/total_x))
    each_y=int(math.ceil(height/total_y))
    #position of starting
    start_x=id_x*each_x
    start_y=id_y*each_y
    #position of ending
    end_x=start_x+each_x
    end_y=start_y+each_y
    #fill this block from starting position to ending position
    for x in range(start_x,end_x):
        if x<=width:
            real = min_x + x * pixel_size_x
            for y in range(start_y,end_y):
                if y<=height:
                    imag = min_y + y * pixel_size_y
                    image[y, x] = mandel(real, imag, iters)
                else:
                    break
        else:
            break
if __name__ == '__main__':
	image = np.zeros((1024, 1536), dtype = np.uint8)
	blockdim = (8, 8)
	griddim = (32, 8)
	#allocate memory for result
	image_global_mem = cuda.to_device(image)
	compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
	image_global_mem.copy_to_host()
	imshow(image)
	show()
