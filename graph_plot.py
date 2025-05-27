import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = 'ep_r_LIA2C_KLD_165000.csv'
data = pd.read_csv(file_path)
y = data['y'].values  # Episode rewards

# Moving average
def moving_avg(data, n):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

# Moving variance
def moving_variance(data, n):
    variances = []
    for i in range(n, len(data)):
        window_data = data[i - n:i]
        var = np.var(window_data)
        variances.append(var)
    return variances

# Window size
window = 1000

# --- Moving average and moving variance ---
avg_data = moving_avg(y, window)
var_data = moving_variance(y, window)
x = [i for i in range(window, len(y))]

# --- Raw (global) variance ---
global_variance = np.var(y)
global_variance_array = [global_variance] * len(y)

# --- Plotting ---

# 1. Moving average
plt.figure()
plt.plot(range(window, len(y)+1), avg_data)
plt.xlabel('Episode')
plt.ylabel('Reward (Moving Avg)')
plt.title('Moving Average of Reward')
plt.tight_layout()
plt.savefig('reward_moving_avg.png')
plt.show()
plt.close()

# 2. Original rewards
plt.figure()
plt.plot(y)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Original Episode Rewards')
plt.tight_layout()
plt.savefig('reward_original.png')
plt.close()

# 3. Moving variance
plt.figure()
plt.plot(x, var_data)
plt.xlabel('Episode')
plt.ylabel('Reward Variance')
plt.title('Moving Variance of Reward')
plt.tight_layout()
plt.savefig('reward_variance.png')
plt.show()
plt.close()

# 4. Raw (global) variance
plt.figure()
plt.plot(global_variance_array, label=f'Global Variance = {global_variance:.2f}')
plt.xlabel('Episode')
plt.ylabel('Reward Variance')
plt.title('Raw Variance of Reward')
plt.legend()
plt.savefig('reward_raw_variance.png')
plt.show()
plt.close()