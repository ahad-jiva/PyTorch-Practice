from temp_convert_func import *

users_unit = input('What unit do you have: ')
if users_unit == ('fahrenheit' or 'Fahrenheit'):
    target_unit = get_target_unit()
    if target_unit == ('celsius' or 'Celsius'):
        input_temp = get_input_temp(users_unit)
        output_temp = far_to_cel(input_temp)
        print_temp(output_temp, target_unit)
    elif target_unit == ('kelvin' or 'Kelvin'):
        input_temp = get_input_temp(users_unit)
        output_temp = far_to_kelv(input_temp)
        print_temp(output_temp, target_unit)
elif users_unit == ('celsius' or 'Celsius'):
    target_unit = get_target_unit()
    if target_unit == ('fahrenheit' or 'Fahrenheit'):
        input_temp = get_input_temp(users_unit)
        output_temp = cel_to_far(input_temp)
        print_temp(output_temp, target_unit)
    elif target_unit == ('kelvin' or 'Kelvin'):
        input_temp = get_input_temp(users_unit)
        output_temp = cel_to_kelv(input_temp)
        print_temp(output_temp, target_unit)
elif users_unit == ('kelvin' or 'Kelvin'):
    target_unit = get_target_unit()
    if target_unit == ('fahrenheit' or 'Fahrenheit'):
        input_temp = get_input_temp(users_unit)
        if float(input_temp) >= 0:
            output_temp = kelv_to_far(input_temp)
            print_temp(output_temp, target_unit)
        else:
            print('You cannot have less than zero kelvin')
    elif target_unit == ('celsius' or 'Celsius'):
        input_temp = get_input_temp(users_unit)
        if float(input_temp) >= 0:
            output_temp = kelv_to_cel(input_temp)
            print_temp(output_temp, target_unit)
        else:
            print('You cannot have less than zero kelvin')
else:
    print('That is not a valid unit')