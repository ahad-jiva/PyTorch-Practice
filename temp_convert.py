def far_to_cel(x):
    far_temp = int(x)
    celsius = (far_temp - 32) / 1.8
    return celsius

def cel_to_far(y):
    cel_temp = int(y)
    fahrenheit = (cel_temp * 1.8) + 32
    return fahrenheit

user_input = input('What unit do you want to convert to: ')
if user_input == ('celsius' or 'Celsius'):
    far_input = int(input('Enter your temperature in fahrenheit: '))
    cel_output = far_to_cel(far_input)
    print(f'That is {cel_output} degrees celsius')
elif user_input == ('fahrenheit' or 'Fahrenheit'):
    cel_input = int(input('Enter your temperature in celsius: '))
    far_output = cel_to_far(cel_input)
    print(f'That is {far_output} degrees fahrenheit')
else:
    print('That is not a valid unit')