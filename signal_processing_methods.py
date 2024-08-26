def count_boxcars(lst):
    count = 0
    in_boxcar = False
    
    for num in lst:
        if num == 1:
            if not in_boxcar:
                count += 1
                in_boxcar = True
        else:
            in_boxcar = False
    
    return count