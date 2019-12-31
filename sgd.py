import math
import numpy as np
import matplotlib.pyplot as plt
import time

#各種データ設定
w = np.array([1.0,0.0]) #パラメータ
x = np.array([
    [1.2,1],[1.25,1],[1.3,1],[1.35,1],
    [1.4,1],[1.45,1],[1.5,1],[1.55,1],
    [1.6,1],[1.65,1],[1.7,1],[1.75,1]
]) #データセット
size_x = len(x) 
y = np.array([0,0,0,0,1,0,1,0,1,1,1,1])
# 試行回数
trial = 100000
# この周期毎にRSMEを保存する
interval = 500
# 学習率
alpha = 0.01
# SGDのデータソートのシード
np.random.seed(114514)
#勾配
g_t = np.array([0,0])
#誤差の初期値と収束判定の閾値
err = 1
th = 1e-12

#データ解析のための中間経過を保存するリスト
mid_rsme = []

#シグモイド関数
def sigm(z):
    return 1 / (1 + np.exp(-1 * z))

def shuffle_samples(arr_a, arr_y):
    zipped = list(zip(arr_a, arr_y))
    np.random.shuffle(zipped)
    a_result, b_result = zip(*zipped)
    return np.asarray(a_result), np.asarray(b_result)    # 型をnp.arrayに変換

for i in range(trial):
    x, y = shuffle_samples(x, y)
    for j in range(size_x):
        #その時点でのパラメータにおける出力を求める
        z = sigm(x[j].dot(w))
        #勾配を求める
        g_t = (z - y[j]) * z * (1 - z) * x[j]
        #パラメータを更新する
        w = w - alpha * g_t

    #収束判定を行う
    zz = sigm(x.dot(w))
    err_tmp = np.sqrt((y-zz).dot(y-zz))
    if (err-err_tmp)**2 < th:
        print(i,":breaking!",w)
        break
    err = err_tmp

    #一定周期毎にRSMEを算出し、保存する。
    zz = sigm(x.dot(w))
    err_tmp = np.sqrt((y-zz).dot(y-zz))
    if (i+1) % interval == 0:
        r = np.sqrt((err_tmp**2) / size_x)
        mid_rsme.append([i+1,r])
        
#最終的な結果を図示する。
xx = np.linspace(1,2,100)
x2 = np.ones((100,2))
x2[:,0] = xx
plt.plot(x2[:,0],sigm(x2.dot(w)))
plt.plot(x[:,0],y,"ro",markersize=10)
plt.hlines([0, 1],1,2,linestyles="dashed")
plt.ylim(-0.1,1.1)
plt.savefig("result_sgd.png")

#別プログラムにてRMSEを解析するためにその値を保存する。
mmm = np.array(mid_rsme)
np.save('mid_rmse_sgd.npy',mmm)