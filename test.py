import numpy as np
import matplotlib.pyplot as plt

load_filename = "mid_rmse.npy"
fig_name = "rsme.png"
#rsmeの値の入ったファイルを開く
file = np.load(load_filename)

#rsmeを図示する。
fig = plt.figure()
plt.ylim(0.3,0.6)
plt.xlabel("trial")
plt.ylabel("RSME")
plt.plot(file[:,0],file[:,1],"ro",markersize=4)
plt.savefig(fig_name)

