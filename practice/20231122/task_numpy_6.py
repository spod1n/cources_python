"""
Task 6. Find sum of diagonal for matrix (5, 5).

Step 1: generate matrix.
Step 2: display diagonal,
Step 3: Find sum of main diagonal,
Step 4: Find alternative diagonal, find elements what more than sum of main diagonal

"""

import numpy as np


print('Neo, you in matrix..')

# task 1
matrix = np.random.randint(1, 10, size=(5, 5))
print('matrix:', matrix, sep='\n', end='\n\n')

# task 2
diagonal = np.diagonal(matrix)
print(f'main diagonal: {diagonal}', end='\n\n')

# task 3
sum_diagonal = np.sum(diagonal)
print(f'sum of main diagonal: {sum_diagonal}', end='\n\n')

# task 4
reverse_matrix = np.fliplr(matrix)
alternative_diagonal = np.diagonal(reverse_matrix)
print(f'alternative diagonal: {alternative_diagonal}', end='\n\n')

elements = alternative_diagonal[alternative_diagonal > sum_diagonal]
print(f'elements what more than sum of main diagonal: {elements}')
