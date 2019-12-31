import numpy as np
import matplotlib.pyplot as plt

load_filename_adam = "mid_rmse.npy"
load_filename_sgd = "mid_rmse_sgd.npy"

fig_name_compare = "rmse_compare.png"
#rsmeの値の入ったファイルを開く
file_adam = np.load(load_filename_adam)
file_sgd = np.load(load_filename_sgd)

#rsmeを図示する。
fig = plt.figure()
plt.ylim(0.3,0.6)
plt.xlabel("trial")
plt.ylabel("RSME")

fig, ax = plt.subplots(facecolor="w")

ax.plot(file_adam[:,0],file_adam[:,1],"ro",markersize=4, label="Adam")
ax.plot(file_sgd[:,0],file_sgd[:,1],"bo",markersize=4, label="SGD")

ax.legend()
plt.savefig(fig_name_compare)
