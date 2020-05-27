import mandelbrot_setup
import numpy as np
from PIL import Image, ImageDraw
import time
import zoom_levels


from numpy           import linspace, reshape
from matplotlib      import pyplot
from multiprocessing import Pool

width, height = 800, 800 # The size of the image created in pixels
zoom_level, file_name = mandelbrot_setup.userInput()

max_iter = zoom_levels.zoom_list[zoom_level]["max_iter"]
# Plot window // Adjust this for panning and zooming // Imaginary and Real parts
# Below is the 'Size' of the coordinate system, decreasing theese will zoom into the coordinatesystem
x_min = zoom_levels.zoom_list[zoom_level]["x_min"]
x_max = zoom_levels.zoom_list[zoom_level]["x_max"]
y_min = zoom_levels.zoom_list[zoom_level]["y_min"]
y_max = zoom_levels.zoom_list[zoom_level]["y_max"]

def mandelbrot_native():
    
    def mandelbrot(c):
        z = 0
        n = 0
        while abs(z) <= 2 and n < max_iter:
            z = z * z + c
            n += 1
        return n

    image = Image.new('HSV', (width, width), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    """ Iterating through each pixel in the image"""
    for x in range(width):
        for y in range(height):
            # Convert pixel coordinate to complex number
            coordinate = complex(x_min + (x / width) * (x_max - x_min),
                                y_min + (y / height) * (y_max - y_min))

            
            # Compute the number of iterations
            m = mandelbrot(coordinate)
            
            color = ()
            """
            if abs(m) > 2:
                v = 765 * m / max_iter
                if v > 510:
                    color = (255, 255, v%255)
                elif v > 255:
                    color = (255, v%255, 0)
                else:
                    color = (v%255, 0, 0)
            else:
                color = (0, 0, 0)
            """

            # The color depends on the number of iterations
            hue = int(255 * m / max_iter)
            saturation = 255
            value = 255 if m < max_iter else 0
            
            
            # Plot the point
            #draw.point([x, y], (int(color[0]), int(color[1]), int(color[2])))
            draw.point([x, y], (hue, saturation, value))


    return image.convert('RGB');
    # image.convert('RGB').save(file_name + '_iterative.png', 'PNG')
    

def mandelbrot_numpy():  
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
            
            color = ()

            
            v = 765 * i / max_iter
            if v > 510:
                color = (255, 255, v%255)
            elif v > 255:
                color = (255, v%255, 0)
            else:
                color = (v%255, 0, 0)

            """
            v = 765 * i / max_iter
            hue = int(v % 255)
            saturation = v % 255
            value = 255 if v < max_iter else abs(i % 256 * 1.5)
            """
            
                
            M[N & (abs(Z) > 2)] = color # Updateing the M matrix if the absolute value is bigger than 2 set the hue
        return M

    M = create_mandelbrot()

    image = Image.fromarray(M, "RGB") # Creating the image from the "M"
    
    rotated_image = _rotate_image(image)
    rotated_image.save(file_name + '_numpy.png', 'PNG') # Saves the image to the current director
    return rotated_image
    

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
    image = render_engine()
    image.save(file_name + render_engine.__name__[10:] + ".png", "PNG") # Saves the image to the current directory



def mandelbrot_multiprocessing():
    def mandelbrot(z): # computation for one pixel
        c = z
        for n in range(max_iter):
            if abs(z)>2: return n  # divergence test
            z = z*z + c
        return max_iter

    X = linspace(x_min,x_max,width) # lists of x and y
    Y = linspace(y_min,y_max,height) # pixel co-ordinates

    # main loops
    p = Pool()
    Z = [complex(x,y) for y in Y for x in X]
    N = p.map(mandelbrot,Z)

    N = reshape(N, (width,height)) # change to rectangular array

    pyplot.imshow(N) # plot the image
    pyplot.show()


times = {}
for re in [mandelbrot_multiprocessing]:
    start = time.time()
    get_mandelbrot(re)
    end = time.time()

    times[re.__name__] = end - start

def time_statestics():
    print(times.get(mandelbrot_native.__name__))
    print(times.get(mandelbrot_numpy.__name__))
    print(times.get(mandelbrot_multiprocessing.__name__))

time_statestics()



