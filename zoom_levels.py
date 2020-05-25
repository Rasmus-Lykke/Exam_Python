original = {
    "x_min": -2.25,
    "x_max": 0.75,
    "y_min": -1.5,
    "y_max": 1.5,
    "max_iter": 250
}
first_level = {
    "x_min": -1.6875,
    "x_max": -0.5625,
    "y_min": -0.5,
    "y_max": 0.5,
    "max_iter": 500
}
second_level = {
    "x_min": -0.22,
    "x_max": -0.21,
    "y_min": -0.70 * 1.12,
    "y_max": -0.69 * 1.12,
    "max_iter": 1000
}
third_level = {
    "x_min": -0.22,
    "x_max": -0.21,
    "y_min": -0.70,
    "y_max": -0.69,
    "max_iter": 1000
}
fourth_level = {
    "x_min": -0.20,
    "x_max": -0.19,
    "y_min": -0.68,
    "y_max": -0.67,
    "max_iter": 1000
}
fifth_level = {
    "x_min": -0.20,
    "x_max": -0.195,
    "y_min": -0.68,
    "y_max": -0.675,
    "max_iter": 1000
}

zoom_list = (original, first_level, second_level, third_level, fourth_level, fifth_level) # Add the dictionaries tho the zoom_list tuple