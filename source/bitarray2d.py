import numpy as np


# a class that behaves like a 2d bool matrix, but packs it's data into a flat bit array
class BitArray2D:
    def __init__(self, size, init_val=False):
        self.size = size
        self.shape = (self.size, self.size)
        if isinstance(init_val, np.ndarray):
            self.array = np.array(init_val, dtype='B', copy=True)
        else:
            self.array = np.zeros(((self.size * self.size) // 8 + 1), dtype='B')
            if init_val:
                self.array += 255

    def __getitem__(self, key):
        (i, j) = key
        ind = (i * self.size + j) // 8
        bit = (i * self.size + j) % 8
        return bool(self.array[ind] & (1 << bit))

    def __setitem__(self, key, value):
        (i, j) = key
        ind = (i * self.size + j) // 8
        bit = (i * self.size + j) % 8
        if value:
            self.array[ind] |= 1 << bit
        elif self.array[ind] & (1 << bit):
            self.array[ind] -= (1 << bit)

    def __and__(self, other):
        if isinstance(other, BitArray2D):
            if other.size != self.size:
                raise ValueError("Could not & two BitArray2Ds with different size")
            else:
                return BitArray2D(self.size, self.array & other.array)
        else:
            raise TypeError("BitArray2D can only & with another BitArray2D")

    def __or__(self, other):
        if isinstance(other, BitArray2D):
            if other.size != self.size:
                raise ValueError("Could not | two BitArray2Ds with different size")
            else:
                return BitArray2D(self.size, self.array | other.array)
        else:
            raise TypeError("BitArray2D can only | with another BitArray2D")

    def get_2d_array(self):
        array_out = np.zeros((self.size, self.size), dtype='bool')
        for i in range(self.size):
            for j in range(self.size):
                array_out[i,j] = self[i,j]
        return array_out
