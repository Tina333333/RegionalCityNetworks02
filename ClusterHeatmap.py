import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster import hierarchy
import ast
import numpy as np

# 加载数据
file_path = 'WCN20240203\\result01.xlsx'
dfOrig = pd.read_csv(file_path)
# ========================================= normalization =============================
df = dfOrig
df = (df - df.min()) / (df.max() - df.min())  # 0-1 normalization

# ======================================= adjusted cosine similarity ==================
cosine_sim = cosine_similarity(df)
cosine_sim_df = pd.DataFrame(cosine_sim, index=dfOrig['Gnodes'], columns=dfOrig['Gnodes'])
# 使用层次聚类，根据相似度重新排列数据
linkage_matrix = hierarchy.linkage(cosine_sim)  # method='single' 'complete' 'average'
order = hierarchy.dendrogram(linkage_matrix, no_plot=True)['leaves']
cosine_sim_df = cosine_sim_df.iloc[order, order]
# 使用sns.clustermap创建聚类热力图
# plt.figure(figsize=(8, 8))
cosine_map = sns.clustermap(cosine_sim_df, cmap='coolwarm', annot=False, linewidths=0,
                            figsize=(8, 8),
                            tree_kws={'cmap': 'coolwarm'})  # Accent
cosine_map.ax_heatmap.tick_params(axis='x', rotation=90)
cosine_map.ax_heatmap.tick_params(axis='y')
# plt.title('Adjusted cosine similarity')
plt.savefig('30WorldCities\\ClusterHeatmapCosine.png')

# ====================================== spearman ======================================

spearman_corr = np.array(df.T.corr(method='spearman'))
spearman_corr_df = pd.DataFrame(spearman_corr, index=dfOrig['Gnodes'], columns=dfOrig['Gnodes'])

# 使用层次聚类，根据相似度重新排列数据
linkage_matrix = hierarchy.linkage(spearman_corr)
order = hierarchy.dendrogram(linkage_matrix, no_plot=True)['leaves']
spearman_corr_df = spearman_corr_df.iloc[order, order]
# 使用sns.clustermap创建聚类热力图
# plt.figure(figsize=(8, 10))
spearman_map = sns.clustermap(spearman_corr_df, cmap='coolwarm', annot=False, linewidths=0,
                              figsize=(8, 8),
                              tree_kws={'cmap': 'coolwarm'})  # Accent
spearman_map.ax_heatmap.tick_params(axis='x', rotation=90)
spearman_map.ax_heatmap.tick_params(axis='y')

# plt.title('Spearman correlation', loc='center')
plt.savefig('30WorldCities\\ClusterHeatmapSpearman.png')
plt.show()
