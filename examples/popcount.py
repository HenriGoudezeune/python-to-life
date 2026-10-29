def popcount(x: u3) -> u2:
    n = 0
    for i in range(3):
        n = n + ((x >> i) & 1)
    return n
