def factorial(x):
    result = 0
    if x == 0:
        result = 1
        return result
    else:
        result = (factorial(x-1)) * x
        return result


def palindrome(string):
    word = [*string]
    if (len(string) == 0) or (len(string) == 1):
        print("That word is a palindrome")
    elif string[0] == string[-1]:
        word.remove(word[0])
        word.remove(word[-1])
        palindrome(word)
    else:
        print("Not a palindrome")

def powers(base, power):
    result = base
    if power == 0:
        return 1
    elif power > 0:
        for i in range(0, power - 1):
            result = result * base
        return result
    else:
        power = power * -1
        for i in range(0, power - 1):
            result = result * base
        return 1/result

