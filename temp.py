import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

# 创建一个假设的不平衡数据集
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=100, random_state=42)

# 可视化原始数据
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1")
plt.title("Original Data Distribution")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# 应用 SMOTETomek
smt = SMOTETomek(smote=SMOTE(k_neighbors=4))
X_res, y_res = smt.fit_resample(X, y)

# 可视化重采样后的数据
plt.subplot(1, 2, 2)
plt.scatter(X_res[y_res == 0][:, 0], X_res[y_res == 0][:, 1], label="Class 0")
plt.scatter(X_res[y_res == 1][:, 0], X_res[y_res == 1][:, 1], label="Class 1")
plt.title("SMOTETomek Resampled Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.show()
