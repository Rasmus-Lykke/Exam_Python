import mandelbrot_setup
import numpy  as np
from PIL import Image, ImageDraw
import time
import zoom_levels
import collections


width, height = 800, 800 # The size of the image created in pixels
zoom_level, file_name = mandelbrot_setup.userInput()

def mandelbrot_native(zoom_level):
    
    max_iter = zoom_levels.zoom_list[zoom_level]["max_iter"]

    def mandelbrot(c):
        z = 0
        n = 0
        while abs(z) <= 2 and n < max_iter:
            z = z * z + c
            n += 1
        return n

    # Plot window // Adjust this for panning and zooming // Imaginary and Real parts
    # Below is the 'Size' of the coordinate system, decreasing theese will zoom into the coordinatesystem
    x_min = zoom_levels.zoom_list[zoom_level]["x_min"]
    x_max = zoom_levels.zoom_list[zoom_level]["x_max"]
    y_min = zoom_levels.zoom_list[zoom_level]["y_min"]
    y_max = zoom_levels.zoom_list[zoom_level]["y_max"]

    image = Image.new('HSV', (width, width), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    """ Iterating through each pixel in the image"""
    for x in range(width):
        for y in range(width):
            # Convert pixel coordinate to complex number
            coordinate = complex(x_min + (x / width) * (x_max - x_min),
                                y_min + (y / width) * (y_max - y_min))

            # Compute the number of iterations
            m = mandelbrot(coordinate)

            # The color depends on the number of iterations
            hue = int(255 * m / max_iter)
            saturation = 255
            value = 255 if m < max_iter else 0

            # Plot the point
            draw.point([x, y], (hue, saturation, value))

    return image.convert('RGB');
    # image.convert('RGB').save(file_name + '_iterative.png', 'PNG')
    

def mandelbrot_numpy(zoom_level):
    max_iter = zoom_levels.zoom_list[zoom_level]["max_iter"] # The number of iterations 

    # Below is the 'Size' of the coordinate system, decreasing theese will zoom into the coordinatesystem
    x_min = zoom_levels.zoom_list[zoom_level]["x_min"]
    x_max = zoom_levels.zoom_list[zoom_level]["x_max"]
    y_min = zoom_levels.zoom_list[zoom_level]["y_min"]
    y_max = zoom_levels.zoom_list[zoom_level]["y_max"]

    

    def create_mandelbrot():
        cmap = lambda value, v_min, v_max, p_min, p_max: p_min + (p_max - p_min) * ((value - v_min) / (v_max - v_min))

        C = np.zeros((width, height), dtype=np.complex_)
        Z = np.zeros((width, height), dtype=np.complex_)
        M = np.zeros((width, height, 3), dtype=np.uint8)

        """ Looping through each pixel given by the height and width variables """
        for cx in range(width):
            for cy in range(height):
                cr = cmap(cx, 0, width, x_min, x_max)
                ci = cmap(cy, 0, height, y_min, y_max)

                C[cx][cy] = cr + ci * 1j

        for i in range(max_iter):
            N = np.less(abs(Z), 2) # Picking out all the elements which are less than 2
            Z[N] = Z[N] * Z[N] + C[N] # Updateing Z, but still only the elements we need to deal with

            # The color depends on the number of iterations
            hue = int(i % 255)
            saturation = i % 255
            value = 255 if i < max_iter else abs(i % 256 * 1.5)
                
            M[N & (abs(Z) > 2)] = [hue, saturation, value] # Updateing the M matrix if the absolute value is bigger than 2 set the hue
        return M

    M = create_mandelbrot()

    image = Image.fromarray(M, "RGB") # Creating the image from the "M"
    
    rotated_image = _rotate_image(image)
    rotated_image.save(file_name + '_numpy.png', 'PNG') # Saves the image to the current director
    return rotated_image
    


def mandelbrot_multiprocessing(zoom_level):
    pass


def _rotate_image(image):
    # rotate the image with expand=True, which makes the canvas
    # large enough to contain the entire rotated image.
    x = image.rotate(90, expand=True)

    # crop the rotated image to the size of the original image
    x = x.crop(box=(x.size[0]/2 - image.size[0]/2,
            x.size[1]/2 - image.size[1]/2,
            x.size[0]/2 + image.size[0]/2,
            x.size[1]/2 + image.size[1]/2))
    return x


def get_mandelbrot(render_engine):
    image = render_engine(zoom_level)
    image.save(file_name + render_engine.__name__[10:] + ".png", "PNG") # Saves the image to the current directory

times = {}

for re in [mandelbrot_native, mandelbrot_numpy]:
    start = time.time()
    get_mandelbrot(re)
    end = time.time()

    times[re.__name__] = end - start

def time_statestics():
    time_difference_sec = times.get(mandelbrot_native.__name__) - times.get(mandelbrot_numpy.__name__)
    print(time_difference_sec)

time_statestics()



