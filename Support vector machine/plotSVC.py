from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

def plot_svc_decision_function(model, ax=None):
    if ax is None:
        #获取当前的子图，如果不存在，则创建新的子图
        ax = plt.gca()  
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    axisx = np.linspace(xlim[0], xlim[1],30)
    axisy = np.linspace(ylim[0], ylim[1],30)
    
    # 其中ravel()是降维函数，vstack能够将多个结构一致的一维数组按行堆叠起来
    axisx, axisy = np.meshgrid(axisx, axisy)
    
    xy=np.vstack([axisx.ravel(), axisy.ravel()]).T
    Z = clf.decision_function(xy).reshape(axisx.shape)
    ax.contour(axisx, axisy, Z,
           colors = "k",
           levels = [-1,0,1],   # 画三条等高线，分别是Z为-1，Z为0和Z为1的三条线
           alpha = 0.5,
           linestyles = ["--", "-", "--"]) 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# 创建50个样本，2个中心
X,y=make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.6)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
# 训练模型
clf = SVC(kernel = "linear").fit(X,y)
plot_svc_decision_function(clf)
plt.show()
    
