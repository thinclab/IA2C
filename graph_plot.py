import matplotlib.pyplot as plt
import os
import pandas as pd
from wheel.macosx_libfile import read_data
from scipy.ndimage.filters import uniform_filter1d
import numpy as np

file_path = 'train_agent_1.csv'
data = pd.read_csv(file_path)
x = data['x']
y = data['y']

x_sliced = x[4635:4754]
y_sliced = y[4635:4754]

def moving_avg(data, n):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

new_data = moving_avg(y, 1000)
print(new_data)
plt.plot(new_data)
plt.xlabel('steps')
plt.ylabel('reward')
plt.savefig('average.png')
plt.show()
plt.close()
plt.plot(y)
plt.xlabel('steps')
plt.ylabel('reward')
plt.savefig('orginal.png')
plt.close()

'''plt.ion()
plt.plot(y)
for i in range(3, len(y), 100):
    y_filter = uniform_filter1d(y, size=i)
    plt.plot(y_filter, '--', label=f"{i}")

#plt.legend()
plt.show()
plt.savefig('test.png')'''
