"""
    A program which constructs three implementations of the mandelbrot set.
    The user needs just to have the required imports installed.
    Then the user have the option of defining the number of processes to be used doing the multiprocessing part.
    Otherwise you just have to initiate the program in the terminal and then following the two steps.
"""

import mandelbrot_setup
import zoom_levels

import numpy as np
from PIL import Image, ImageDraw
import time

from numba import jit
from itertools import repeat
from multiprocessing import Pool  # For parallel execution of a function

processes = 4  # Change this to alter the performance og the multiprocessing function.
# For me 4 is the optimal with 6 processing cores

times = {}  # Dictionary for holding the times
width, height = 800, 800  # The size of the image created in pixels
zoom_level, file_name = mandelbrot_setup.user_input()  # Calling the user_input() function the in setup file and
# assigning the variables to the return values.

max_iter = zoom_levels.zoom_list[zoom_level]["max_iter"]

# Below is the 'Size' of the coordinate system, decreasing these will zoom into the coordinate system
x_min = zoom_levels.zoom_list[zoom_level]["x_min"]
x_max = zoom_levels.zoom_list[zoom_level]["x_max"]
y_min = zoom_levels.zoom_list[zoom_level]["y_min"]
y_max = zoom_levels.zoom_list[zoom_level]["y_max"]


def mandelbrot_naive():
    def mandelbrot(c):
        z = 0
        n = 0
        while abs(z) <= 2 and n < max_iter:
            z = z * z + c
            n += 1
        return n

    image = Image.new('HSV', (width, width), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Iterating through each pixel in the image
    for ix in range(width):
        for iy in range(height):
            # Convert pixel coordinate to complex number
            coordinate = complex(x_min + (ix / width) * (x_max - x_min),
                                 y_min + (iy / height) * (y_max - y_min))

            # Compute the number of iterations
            m = mandelbrot(coordinate)

            # The color depends on the number of iterations
            hue = int(255 * m / max_iter)
            saturation = 255
            value = 255 if m < max_iter else 0

            # Draw the point
            draw.point([ix, iy], (hue, saturation, value))

    return image.convert('RGB')


def mandelbrot_numpy():
    def create_mandelbrot():
        """ Lambda function / anonymous function which calculates the cmap """
        cmap = lambda value, v_min, v_max, p_min, p_max: p_min + (p_max - p_min) * ((value - v_min) / (v_max - v_min))

        C = np.zeros((width, height), dtype=np.complex_)
        Z = np.zeros((width, height), dtype=np.complex_)
        result = np.zeros((width, height, 3), dtype=np.uint8)

        # Looping through each pixel given by the height and width variables
        for cx in range(width):
            for cy in range(height):
                cr = cmap(cx, 0, width, x_min, x_max)
                ci = cmap(cy, 0, height, y_min, y_max)

                C[cx][cy] = cr + ci * 1j

        for i in range(max_iter):
            N = np.less(abs(Z), 2)  # Picking out all the elements which are less than 2
            Z[N] = Z[N] * Z[N] + C[N]  # Updating Z, but still only the elements we need to deal with

            # The color depends on the number of iterations
            v = 765 * i / max_iter
            if v > 510:
                color = (255, 255, v % 255)
            elif v > 255:
                color = (255, v % 255, 0)
            else:
                color = (v % 255, 0, 0)

            result[N & (abs(Z) > 2)] = color  # Updating the M matrix if the absolute value is bigger than 2 set the hue
        return result

    M = create_mandelbrot()

    image = Image.fromarray(M, "RGB")  # Creating the image from the "M"

    rotated_image = _rotate_image(image)
    rotated_image.save(file_name + '_numpy.png', 'PNG')  # Saves the image to the current directory
    return rotated_image

""" This code is mainly found from the following link and are thereafter fitted to the rest of the program 
https://medium.com/convergence-tech/visualize-the-mandelbrot-set-in-gigapixels-python-15e4ad459925 """
def mandelbrot_multiprocessing():
    @jit  # Using Numba to translate the function to optimized machine code at runtime this avoids the pickle error
    def get_col(args):
        iy, width, height = args
        result = np.zeros((1, width, 3))
        for ix in np.arange(width):

            x0 = x_min + (ix / width) * (x_max - x_min)
            y0 = y_min + (iy / height) * (y_max - y_min)

            x = 0.0
            y = 0.0

            for i in range(max_iter):
                x_new = x * x - y * y + x0
                y = 2 * x * y + y0
                x = x_new

                if abs(x) > 2:
                    # The color depends on the number of iterations
                    v = 765 * i / max_iter
                    if v > 510:
                        color = (255, 255, v % 255)
                    elif v > 255:
                        color = (255, v % 255, 0)
                    else:
                        color = (v % 255, 0, 0)

                    result[0, ix] = color

        return result

    result = np.zeros((height, width, 3), dtype=np.uint8)
    iy = np.arange(height)

    pool = Pool(processes)  # 4 = Number of processes
    mandelbrot = pool.map_async(get_col, zip(iy, repeat(width), repeat(height))).get()
    pool.close()
    pool.join()

    for ix in np.arange(height):
        result[ix, :] = mandelbrot[ix]

    mandelbrot = result
    mandelbrot = np.clip(mandelbrot * 255, 0, 255).astype(np.uint8)
    mandelbrot = Image.fromarray(mandelbrot)

    return mandelbrot


def _rotate_image(image):
    """ Used for rotating the numpy image and cropping to avoid a wide and low image.
    The function is found online but i have lost the source"""

    # Rotate the image with expand=True, which makes the canvas
    # large enough to contain the entire rotated image.
    x = image.rotate(90, expand=True)

    # crop the rotated image to the size of the original image
    x = x.crop(box=(x.size[0] / 2 - image.size[0] / 2,
                    x.size[1] / 2 - image.size[1] / 2,
                    x.size[0] / 2 + image.size[0] / 2,
                    x.size[1] / 2 + image.size[1] / 2))
    return x


def timer(represent):
    """ Decorator function """
    def wrapper():
        """ Looping through the three rendering engines and passes the functions to the get_mandelbrot function and
        then calling the timer_statistics function fore a representation of the time difference"""
        for re in [mandelbrot_naive, mandelbrot_numpy, mandelbrot_multiprocessing]:
            start = time.time()
            get_mandelbrot(re)  # Calls the function and parses the rendering_engine as a parameter
            end = time.time()

            times[re.__name__] = end - start

        represent()  # Time statistics function

        print("Done!")

    return wrapper


@timer
def time_statistics():
    """ Used for representing the time difference between the three functions """
    time_naive = times.get(mandelbrot_naive.__name__)
    time_numpy = times.get(mandelbrot_numpy.__name__)
    time_multiprocessing = times.get(mandelbrot_multiprocessing.__name__)

    print()

    print(f'Naive:            {str(time_naive)[:6]} sec.')

    print(f'Numpy:            {str(time_numpy)[:6]} sec. '
          f'| Time difference: {str(time_numpy - time_naive)[:6]} sec. '
          f'| {str(((time_naive - time_numpy) / ((time_naive + time_numpy) / 2)) * 100)[:5]} % ')

    print(f'Multiprocessing:  {str(time_multiprocessing)[:6]} sec. '
          f'| Time difference: {str(time_multiprocessing - time_numpy)[:6]} sec. '
          f'| {str(((time_numpy - time_multiprocessing) / ((time_numpy + time_multiprocessing) / 2)) * 100)[:5]} %')


def get_mandelbrot(render_engine):
    """ High order function which takes a rendering_engine as is parameter and saves the image created by the
    rendering engine """
    # Calls the function parsed in the parameter and assigns the variable "image" to the return value
    image = render_engine()
    image.save(file_name + render_engine.__name__[10:] + ".png", "PNG")  # Saves the image to the current directory
    print(f'File: {render_engine.__name__} saved')


time_statistics()
