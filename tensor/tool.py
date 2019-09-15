import random
rand = random.Random()
rand2 = random.Random()

def set_randomly(max_range, out_array):
    for i in range(len(out_array)):
        out_array[i] = rand.randint(0, max_range)

def set_shuffle(out_array):
    rand2.shuffle(out_array)

def copy(source_array, start, length, out_array):
    for i in range(length):
        out_array[i] = source_array[start + i]



def copy_row(source_array, source_shape, point_array, out_array):
    column = source_shape[-1]
    source_row = len(source_array) // column
    out_row = len(out_array) // column

    for r in range(out_row):
        choice = point_array[r]
        for c in range(column):
            out_array[r * column + c] = source_array[choice * column + c]
