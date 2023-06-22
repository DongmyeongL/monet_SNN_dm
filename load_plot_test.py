import numpy as np
import matplotlib.pyplot as plt;


spike_data=np.loadtxt('./ptest.txt');

plt.scatter(spike_data[:,1],spike_data[:,0],s=1)

plt.show();
