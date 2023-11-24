import numpy as np

"""Task 1. Create list with 50 elements, reshape in 3 dim"""
arr50 = np.arange(1, 51, dtype=np.uint64)
print(arr50.reshape(2, 5, 5), arr50.reshape(10, 5), arr50.reshape(25, 2), sep='\n\n\n')


"""Task 2. Find elements, where elements more than 20 and less 30"""
arr50 = np.arange(1, 51, dtype=np.uint64)
print(arr50[(arr50 > 20) & (arr50 < 30)])


"""Task 3. Find elements, where elements is even"""
arr50 = np.arange(1, 51, dtype=np.uint64)
print(arr50[arr50 % 2 == 0])


"""Task 4. Generate chess dashboard """
size = 8
chessboard = np.zeros((size, size), dtype=np.uint64)
chessboard[1::2, ::2] = 1
chessboard[::2, 1::2] = 1
print(chessboard)


"""Task 5. Find elements, where elements is NOT even, less than 41 and more then 15 and elements not equal 21"""
arr50 = np.arange(1, 51, dtype=np.uint64)
print(arr50[~(arr50 % 2 == 0) & (arr50 < 41) & (arr50 > 15) & (arr50 != 20)])