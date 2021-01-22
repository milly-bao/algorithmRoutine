#=========================sklearn数据库=========================================
#%%
#标准数据集调取-波士顿房价-回归模型
from sklearn.datasets import load_boston
boston = load_boston() #调取数据集
boston.data.shape #数据集格式,nparray数组


#%%
#标准模型调取-鸢尾花-多分类
from sklearn.datasets import load_iris
iris = load_iris()
iris.data.shape
iris.target_names


#%%
#无监督聚类KMeans
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

fileContent = pd.read_excel('/Users/milly/Desktop/文件/寒假/机器学习/1999年全国31个省份城镇居民家庭平均每人全年消费性支出.xlsx')
data,cityName = [],[]
for i,city in fileContent.iterrows():
    cityName.append(city['城市'])
    data.append([float(city['食品']),float(city['食品'])])
km = KMeans(n_clusters=3) #n_clusters:目标类别数
label = km.fit_predict(data) #每条数据分别所属类型已得出,默认使用欧式距离
expenses = np.sum(km.cluster_centers_,axis=1) #对每个类别求均值
CityCluster = [[],[],[]]
for i in range(len(cityName)):
    CityCluster[label[i]].append(cityName[i])
for i in range(len(CityCluster)):
    print("Expenses:%.2f" % expenses[i])
    print(CityCluster[i])
    
    
#%%
#无监督学习聚类DBSCAN
import numpy as np
import pandas as pd
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

fileContent = pd.read_excel('/Users/milly/Desktop/文件/寒假/机器学习/学生上网数据编.xlsx')
mac2id = {}
onlinetimes = []
for i,mac in fileContent.iterrows():
    m = mac['MAC地址']
    v = (mac['开始上网时间'],mac['上网时间'])
    onlinetimes.append(v)
    mac2id[m] = v
real_X=np.array(onlinetimes).reshape((-1,2)) #-1用于行数不知道的情况   
X=real_X[:,0:1]
db=skc.DBSCAN(eps=2,min_samples=2).fit(X) #参数eps为最接近距离，min_sample为最小个数
labels = db.labels_

print('Labels:') #逐个打出标签
print(labels)
raito=len(labels[labels[:] == -1]) / len(labels)
print('Noise raito:',format(raito, '.2%')) #计算为噪音的

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_) #总类别数
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels)) #聚类效果评价指标

for i in range(n_clusters_):
    print('Cluster ',i,':')
    print(list(X[labels == i].flatten()))
    
plt.hist(X,24)


#%%
#无监督学习降维-PCA-常用语多维度数据的降维
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris() #以字典形式加载鸢尾花数据集
y = data.target #使用y表示数据集中的标签
X = data.data #使用X表示数据集中的属性数据
pca = PCA(n_components=2) #加载PCA算法，设置降维后主成分数目为2
reduced_X = pca.fit_transform(X) #对原始数据进行降维，保存在reduced_X中, 降维后的属性数据
#可视化
red_x, red_y = [], [] #第一类数据点
blue_x, blue_y = [], [] #第二类数据点
green_x, green_y = [], [] #第三类数据点
for i in range(len(y)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    elif y[i] == 2:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
plt.scatter(red_x, red_y, c='r', marker='x') #第一类数据点
plt.scatter(blue_x, blue_y, c='b', marker='D') #第二类数据点
plt.scatter(green_x, green_y, c='g', marker='.') #第三类数据点
plt.show()


#%%
#无监督学习降维-NMF-常用语图片信息的提取降维
import matplotlib.pyplot as plt #加载matplotlib用于数据的可视化
from sklearn import decomposition #加载PCA算法包
from sklearn.datasets import fetch_olivetti_faces #加载Olivetti人脸数据集导入函数
from numpy.random import RandomState #加载RandomState用于创建随机种子

n_row, n_col = 2, 3 #设置图像展示时的排列情况
n_components = n_row * n_col #设置提取的特征的数目
image_shape = (64, 64) #设置人脸数据图片的大小
dataset = fetch_olivetti_faces(shuffle=True,random_state=RandomState(0))
faces = dataset.data

def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2.*n_col, 2.26*n_row)) #设置图片大小
    plt.suptitle(title, size=16) #设置标题
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i+1)
        vmax = max(comp.max(),-comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray, 
                   interpolation='nearest', vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0)
    
estimators = [('Eigenfaces - PCA using randomized SVD',
               decomposition.PCA(n_components=6,whiten=True)),
    ('Non-negative components - NMF',
     decomposition.NMF(n_components=6, init='nndsvda', tol=5e-3))]
for name, estimator in estimators: #分别调用PCA和NMF
    estimator.fit(faces) #调用PCA或NMF提取特征
    components_ = estimator.components_ #获取提取的特征
    plot_gallery(name, components_[:n_components])
plt.show()































