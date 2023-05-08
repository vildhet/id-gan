

def is_power_of_two(n):
    if n <= 0:
        return False
    else:
        return (n & (n - 1)) == 0
