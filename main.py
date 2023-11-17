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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN, SMOTETomek

# 设置是否使用重采样的开关
use_resampling = True  # 将此设置为False以关闭重采样

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
    "RandomForest": RandomForestClassifier(),
}

if use_resampling:
    param_grids = {
        "RandomForest": {
            "RandomForest__n_estimators": [50, 100, 150],
            "RandomForest__max_depth": [3, 5, 10, None],
            "RandomForest__min_samples_split": [2, 4, 6],
            "RandomForest__min_samples_leaf": [1, 2, 4],
            "RandomForest__bootstrap": [True, False],
            "RandomForest__class_weight": ["balanced", None]
        },
    }
else:
    param_grids = {
        "RandomForest": {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
            "class_weight": ["balanced", None]
        },
    }

## 数据处理和模型训练
cv = StratifiedKFold(n_splits=5)
best_models = {}
best_scores = {}

n_neighbors = 3

for model_name, model in models.items():
    if use_resampling:
        resampling_strategies = {
            "ADASYN": ADASYN(n_neighbors=n_neighbors),
            "SMOTE": SMOTE(k_neighbors=n_neighbors),
            "SMOTEENN": SMOTEENN(smote=SMOTE(k_neighbors=min(n_neighbors, len(y_train) // 2))),
            "SMOTETomek": SMOTETomek(smote=SMOTE(k_neighbors=min(n_neighbors, len(y_train) // 2)))
        }
    else:
        resampling_strategies = {"None": None}  # 不使用重采样

    for sampler_name, sampler in resampling_strategies.items():
        if sampler is None:
            estimator = model
        else:
            estimator = Pipeline([(sampler_name, sampler), (model_name, model)])

        grid_search = GridSearchCV(estimator, param_grids[model_name], cv=cv, scoring='roc_auc', verbose=3)
        grid_search.fit(X_train, y_train)

        best_models[f"{model_name}_{sampler_name if sampler else 'NoResampling'}"] = grid_search.best_estimator_
        best_scores[f"{model_name}_{sampler_name if sampler else 'NoResampling'}"] = grid_search.best_score_

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
