import random

def guessing_game(x):
    random_number = random.randint(0, x)
    guess = 0
    while guess != random_number:
        guess = int(input('Guess the mystery number: '))
        if guess == random_number:
            print('Congrats, you guessed the number')
            guess = random_number
        elif guess > random_number:
            print('Nope, your guess is too big')
        elif guess < random_number:
            print('Nope, your guess is too small')

guessing_game(100)