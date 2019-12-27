import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

#各種データ設定
w = np.array([1.0,0.0]) #パラメータ
x = np.array([
    [1.2,1],[1.25,1],[1.3,1],[1.35,1],
    [1.4,1],[1.45,1],[1.5,1],[1.55,1],
    [1.6,1],[1.65,1],[1.7,1],[1.75,1]
]) #データセット
size_x = len(x) 
y = np.array([0,0,0,0,1,0,1,0,1,1,1,1]) #分類表
trial = 50000
interval = 500
alpha = 0.01
np.random.seed(114514)

ims = []
fig = plt.figure()
ranges = np.arange(0,2,0.01)

#データ解析のための中間経過を保存するリスト
mid_w = np.zeros((400,2))
mid_rsme = np.zeros((400,2))

#シグモイド関数
def f(j):
    return 1 / (1 + math.exp(-1 * (w[0] * x[j,0] + w[1] * x[j,1])))

#シグモイド関数(パラメータ指定)
def f_range(j_array):
    y_arr = np.zeros(len(j_array))
    for i in range(len(j_array)):
        y_arr[i] = 1 / (1 + math.exp(-1 * (w[0] * j_array[i] + w[1])))
    return y_arr

#RMSEの算出
def rmse():
    res = 0
    for j in range(size_x):
        res += (y[j] - f(j))**2
    res /= size_x
    return np.sqrt(res)

def shuffle_samples(arr_a, arr_y):
    zipped = list(zip(arr_a, arr_y))
    np.random.shuffle(zipped)
    a_result, b_result = zip(*zipped)
    return np.asarray(a_result), np.asarray(b_result)    # 型をnp.arrayに変換

for i in range(trial):
    x, y = shuffle_samples(x, y)
    for j in range(size_x):
        z = f(j)
        g_t = (z - y[j]) * z * (1 - z) * x[j]
        w = w - alpha * g_t
    if (i+1) % interval == 0:
        p = int((i+1)/interval) - 1
        r = rmse()
        mid_w[p,0] = w[0]
        mid_w[p,1] = w[1]
        mid_rsme[p] = np.array([(i+1)/125,r])
        im = plt.plot(ranges, f_range(ranges), color='red')
        im.append(plt.scatter(x[:,0], y, color='red'))
        im.append(plt.text(0.1, 0.8, r'trial: ' + str(i + 1)))
        ims.append(im)
        
#解析のためのデータ保存
np.save('mid_w_sgd.npy',mid_w)
np.save('mid_rmse_sgd.npy',mid_rsme)
ani = animation.ArtistAnimation(fig, ims, interval=100)
ani.save('anim.mp4', writer="ffmpeg")
plt.show()