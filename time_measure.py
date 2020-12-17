import os

import numpy as np

num_cores_array = np.arange(1, 17, 1)
for num_cores in num_cores_array:
    command1 = f'mpirun -n {num_cores} python train.py'
    os.system(command1)
    # then we will have file with time measure
    command2 = f'mv outputs/voc_100.txt outputs/voc_100.cores_{num_cores}.txt'
    os.system(command2)