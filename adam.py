import math
import numpy as np
import matplotlib.pyplot as plt

#各種データ設定
w = np.array([1.0,0.0])
x = np.array([
	[1.2,1],[1.25,1],[1.3,1],[1.35,1],
	[1.4,1],[1.45,1],[1.5,1],[1.55,1],
	[1.6,1],[1.65,1],[1.7,1],[1.75,1]
])
size_x = len(x)
y = np.array([0,0,0,0,1,0,1,0,1,1,1,1])
#試行回数
trial = 100000
#この周期毎にRSMEを保存する
interval = 500
#学習率
alpha = 0.001
#各モーメントの減衰率
beta1 = 0.9
beta2 = 0.999
#0除算を防ぐための微小な値
epsilon = 0.00000001
#モーメントベクトルの値
m = np.array([0,0])
v = np.array([0,0])
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

for i in range(trial):
	for j in range(size_x):
		#その時点でのパラメータにおける出力を求める
		z = sigm(x[j].dot(w))
		#勾配を求める
		g_t = (z - y[j]) * z * (1 - z) * x[j]
		#各モーメントベクトルを求める
		m = beta1 * m + (1 - beta1) * g_t
		v = beta2 * v + (1 - beta2) * g_t * g_t
		#各モーメントベクトルにバイアス補正をかける
		mt = m / (1 - (beta1**(i+1)))
		vt = v / (1 - (beta2**(i+1)))
		#パラメータを更新する
		w = w - alpha * (m / (np.sqrt(vt)+epsilon))
	#収束判定を行う
	zz = sigm(x.dot(w))
	err_tmp = np.sqrt((y-zz).dot(y-zz))
	if (err-err_tmp)**2 < th:
		print(i,":breaking!")
		break
	err = err_tmp
	
	#一定周期毎にRSMEを算出し、保存する。
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
plt.savefig("aaa.png")

#別プログラムにてRMSEを解析するためにその値を保存する。
mmm = np.array(mid_rsme)
np.save('mid_rmse.npy',mmm)