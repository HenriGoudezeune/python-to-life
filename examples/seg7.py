def seg7(d: u4) -> u7:
    """7-segment decoder: which segments light up for digit d (0-9).

    Bit 0..6 = segment a..g:   aaa
                              f   b
                               ggg
                              e   c
                               ddd
    Large in Life: about 230 gates, 360 gliders and 1300 still lifes.
    """
    s = 0
    if d == 0:
        s = 0b0111111
    elif d == 1:
        s = 0b0000110
    elif d == 2:
        s = 0b1011011
    elif d == 3:
        s = 0b1001111
    elif d == 4:
        s = 0b1100110
    elif d == 5:
        s = 0b1101101
    elif d == 6:
        s = 0b1111101
    elif d == 7:
        s = 0b0000111
    elif d == 8:
        s = 0b1111111
    elif d == 9:
        s = 0b1101111
    return s
