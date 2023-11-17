import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from joblib import dump, load
import matplotlib.pyplot as plt
# import wandb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. 数据加载与预处理
file_path = 'data_DM.xlsx'
df = pd.read_excel(file_path)

## 选择特征和重新定义标签为二分类问题
features = df.drop(["患病时间-10-15-", "患病时间-5-10-", "TBA", "PCT", "PLCR", "AFU"], axis=1)
labels = df["患病时间-10-15-"].replace({1: 0, 2: 1, 3: 1})

## 数据预处理
features.fillna(features.median(), inplace=True)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

## 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)

# 2. 模型训练与选择
models = {
    # "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    # "SVM": SVC(probability=True),
    # "BP_NeuralNetwork": MLPClassifier(),
    # "XGBoost": XGBClassifier(verbosity=1),
    # "LightGBM": LGBMClassifier(),
    # "CatBoost": CatBoostClassifier()
}

param_grids = {
    # "DecisionTree": {
    #     "DecisionTree__max_depth": [3, 4, 5, 6, None],
    #     "DecisionTree__min_samples_split": [2, 4, 6, 8],
    #     "DecisionTree__min_samples_leaf": [1, 2, 3, 4],
    #     "DecisionTree__criterion": ["gini", "entropy"],
    #     "DecisionTree__splitter": ["best", "random"],
    #     "DecisionTree__class_weight": ["balanced"]  # 添加类别权重
    # },
    "RandomForest": {
        "RandomForest__n_estimators": [50, 100, 150],  # 增加树的数量
        "RandomForest__max_depth": [3, 5, 10, None],  # 考虑更小的深度以及无限制深度
        "RandomForest__min_samples_split": [2, 4, 6],  # 调整分裂所需最小样本数
        "RandomForest__min_samples_leaf": [1, 2, 4],  # 调整叶节点的最小样本数
        "RandomForest__bootstrap": [True, False],  # 是否使用Bootstrap抽样
        "RandomForest__class_weight": ["balanced", None]  # 考虑类别权重
    },
    # "SVM": {"SVM__C": [0.1, 1, 10], "SVM__kernel": ["linear", "rbf"]},
    # "BP_NeuralNetwork": {"BP_NeuralNetwork__hidden_layer_sizes": [(50,), (100,), (50, 50)],
    #                      "BP_NeuralNetwork__activation": ["tanh", "relu"]},
    # "XGBoost": {
    #     "XGBoost__n_estimators": [50, 100, 200],  # 增加树的数量
    #     "XGBoost__learning_rate": [0.01, 0.05, 0.1, 0.2],  # 更详细的学习率设置
    #     "XGBoost__max_depth": [3, 4, 6],  # 调整最大深度
    #     "XGBoost__min_child_weight": [1, 2, 4],  # 最小子节点样本权重和
    #     "XGBoost__gamma": [0, 0.1, 0.2],  # 节点分裂所需的最小损失函数下降值
    #     "XGBoost__subsample": [0.6, 0.8, 1.0],  # 训练每棵树时抽取的样本比例
    #     "XGBoost__colsample_bytree": [0.6, 0.8, 1.0],  # 在建立树时对特征进行采样的比例
    #     "XGBoost__reg_lambda": [1, 1.5, 2],  # L2正则化项
    #     "XGBoost__reg_alpha": [0, 0.1, 0.2]  # L1正则化项
    # },

    # "XGBoost": {
    #     "XGBoost__n_estimators": [100],  # 较少的树的数量
    #     "XGBoost__learning_rate": [0.05],  # 更详细的学习率设置
    #     "XGBoost__max_depth": [3],  # 调整最大深度
    #     "XGBoost__min_child_weight": [1, 2, 4],  # 默认值
    #     "XGBoost__gamma": [0, 0.1, 0.2],  # 轻微的分裂损失阈值
    #     "XGBoost__subsample": [0.8],  # 子样本比例
    #     "XGBoost__colsample_bytree": [0.8],  # 特征采样比例
    #     "XGBoost__reg_lambda": [1, 1.5],  # L2正则化
    #     "XGBoost__reg_alpha": [0, 0.1]  # L1正则化
    # }


    # "LightGBM": {
    #     "LightGBM__n_estimators": [50, 100, 150],
    #     "LightGBM__learning_rate": [0.01, 0.1, 0.2],
    #     "LightGBM__max_depth": [3, 4, 5],
    #     "LightGBM__reg_lambda": [1, 2, 3]
    # },
    # "CatBoost": {
    #     "CatBoost__iterations": [50, 100, 150],
    #     "CatBoost__learning_rate": [0.01, 0.1, 0.2],
    #     "CatBoost__depth": [3, 4, 5],
    #     "CatBoost__l2_leaf_reg": [1, 2, 3]  # 正则项
    # }
}

## 数据处理和模型训练
cv = StratifiedKFold(n_splits=5)
best_models = {}
best_scores = {}

## 使用WandB记录
# wandb.init(project="diabetes_prediction")

n_neighbors = 3

for model_name, model in models.items():
    for sampler_name, sampler in {"ADASYN": ADASYN(n_neighbors=n_neighbors),
                                  "SMOTE": SMOTE(k_neighbors=n_neighbors)}.items():
        pipeline = Pipeline([(sampler_name, sampler), (model_name, model)])
        grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=cv, scoring='roc_auc', verbose=3)
        grid_search.fit(X_train, y_train)

        best_models[f"{model_name}_{sampler_name}"] = grid_search.best_estimator_
        best_scores[f"{model_name}_{sampler_name}"] = grid_search.best_score_

        # wandb.log({f"{model_name}_{sampler_name}_best_score": grid_search.best_score_})

# 3. 保存最佳模型
## 输出最佳模型和分数
print("the best model is: ", best_models)
print("the best scores are: ", best_scores)
best_model_name = max(best_scores, key=best_scores.get)
best_model = best_models[best_model_name]
dump(best_model, 'best_model.joblib')

## 模型预评估
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {name}')
    plt.legend(loc="lower right")
    plt.show()

# 4. 加载测试数据并预处理
test_file_path = 'data_DM_test.xlsx'
df_test_origin = pd.read_excel(test_file_path)
df_test = df_test_origin.drop(["患病时间-5-10-", "LDH1", "MG", "PCT", "PLCR", "AFU"], axis=1)
df_test = df_test.dropna()
test_features = df_test.drop(["患病时间-10-15-"], axis=1)
test_labels = df_test["患病时间-10-15-"].replace({1: 0, 2: 1, 3: 1})
test_features_scaled = scaler.transform(test_features)

# 5. 评估模型性能
loaded_model = load('best_model.joblib')
y_pred_test = loaded_model.predict(test_features_scaled)
y_prob_test = loaded_model.predict_proba(test_features_scaled)[:, 1]
print("Classification Report on Test Data:")
print(classification_report(test_labels, y_pred_test))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, y_pred_test))
fpr_test, tpr_test, _ = roc_curve(test_labels, y_prob_test)
roc_auc_test = auc(fpr_test, tpr_test)
plt.figure()
plt.plot(fpr_test, tpr_test, label=f'Test ROC curve (area = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on Test Data')
plt.legend(loc="lower right")
plt.show()
