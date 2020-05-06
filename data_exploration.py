import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

features = ["LB", "AC", "FM", "UC", "ASTV", "MSTV", "ALTV", "MLTV", "DL"]
data = pd.read_excel("CTG.xls", sheet_name="Raw Data")
data = data.dropna()
data = data.loc[:, features]
normalised_features = StandardScaler().fit_transform(data)

#constant_rows = data[data.apply(lambda x: min(x) == max(x), 1)]
#print(constant_rows)

#duplicate_rows = data[data.duplicated()]
#print(len(duplicate_rows))

#pca = PCA(n_components=len(features)).fit(normalised_features)

#for f, p in zip(features, pca.explained_variance_ratio_):
    #print(f + ": %.2f%%" % float(p*100))

#sns.pairplot(data)
#plt.show()

z = np.abs(stats.zscore(data))
print(len(np.where(z > 3)[0]))