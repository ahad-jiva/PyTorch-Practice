def far_to_cel(x):
    far_temp = float(x)
    celsius = (far_temp - 32) / 1.8
    return celsius

def far_to_kelv(z):
    far_temp = float(z)
    kelvin = ((far_temp - 32) * 5/9) + 273.15
    return kelvin

def cel_to_far(y):
    cel_temp = float(y)
    fahrenheit = (cel_temp * 1.8) + 32
    return fahrenheit

def cel_to_kelv(w):
    cel_temp = float(w)
    kelvin = cel_temp + 273.15
    return kelvin

def kelv_to_cel(u):
    kelv_temp = float(u)
    celsius = kelv_temp - 273.15
    return celsius

def kelv_to_far(t):
    kelv_temp = float(t)
    fahrenheit = ((kelv_temp) - 273.15) * (9/5) +32
    return fahrenheit

def get_input_temp(unit):
    input_temp = input(f'Enter your temperature in {unit}: ')
    return input_temp

def get_target_unit():
    target_unit = input('What unit do you want to convert this to: ')
    return target_unit

def print_temp(temp, unit):
    print(f'That is {temp} degrees {unit}.')