from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from util import evaluate_and_plot


# 对新的数据集应用同样的预处理和特征选择
def preprocess_and_select_features(x_new, scaler_, indices_):
    # 应用标准化
    x_new_scaled = scaler_.transform(x_new)
    # 应用特征选择
    x_new_selected = x_new_scaled[:, indices_]
    return x_new_selected


# def combined_prediction(model1, model2, x_test_1, x_test_2):
#     """
#     使用两个模型进行预测，其中model1用于预测类别0，model2用于预测类别1。
#     """
#     # 使用两个模型分别进行预测
#     pred1 = model1.predict(x_test_1)
#     pred2 = model2.predict(x_test_2)
#
#     # 合并预测结果
#     combined_pred = np.where(pred1 == 0, 0, pred2)
#
#     return combined_pred

def combined_prediction(model1, model2, x_test_1, x_test_2, weight_ratio=2):
    """
    组合两个模型的预测结果。

    如果两个模型都预测为 0，则最终预测为 0。
    如果两个模型都预测为 1，则最终预测为 1。
    其他情况下，根据模型的预测概率和指定的权重比例进行预测。

    :param model1: 第一个模型
    :param model2: 第二个模型
    :param x_test_1: 测试集特征数据_model1
    :param x_test_2: 测试集特征数据_model2
    :param weight_ratio: 权重比例，默认为2，表示 model1 的权重是 model2 的两倍
    :return: 组合预测结果
    """

    # 获取两个模型的预测结果
    pred1 = model1.predict(x_test_1)
    pred2 = model2.predict(x_test_2)

    # 获取两个模型的预测概率
    proba1 = model1.predict_proba(x_test_1)[:, 1]
    proba2 = model2.predict_proba(x_test_2)[:, 1]

    # 组合预测
    combined_pred = []
    for i in range(len(pred1)):
        if pred1[i] == pred2[i]:
            combined_pred.append(pred1[i])
        else:
            # 按照给定的权重比例计算最终概率
            combined_proba = (weight_ratio * proba1[i] + proba2[i]) / (weight_ratio + 1)
            # combined_pred.append(1 if combined_proba > 0.6 else 0)
            if (proba2[i] < 0.85 and proba1[i] < 1e-8) or (proba1[i] > 0.1 and 0.997 > proba2[i] > 0.99):
                combined_pred.append(1)
            else:
                combined_pred.append(0)
    return np.array(combined_pred)


def evaluate_and_plot_2(model1, model2, x_test_1, x_test_2, y_test_, title):
    # 结合两个模型的预测结果
    y_pred_test = combined_prediction(model1, model2, x_test_1, x_test_2)

    # 打印分类报告和混淆矩阵
    print(f"Classification Report for {title}:")
    print(classification_report(y_test_, y_pred_test, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_, y_pred_test))

    # 如果需要绘图，您可以在这里添加相应的绘图代码


if __name__ == '__main__':
    # 加载标准化处理器
    scaler_1 = load('35-0-6-43/scaler.joblib')
    scaler_2 = load('20-15-1-48/scaler.joblib')
    # 加载特征选择的索引
    indices_1 = load('35-0-6-43/indices.joblib')
    indices_2 = load('20-15-1-48/indices.joblib')

    a1 = np.sort(indices_1)
    a2 = np.sort(indices_2)
    # 加载训练好的模型
    model_1 = load('35-0-6-43/temp.joblib')
    model_2 = load('20-15-1-48/temp.joblib')

    file_path = 'test_data.xlsx'
    test_df = pd.read_excel(file_path)

    y_test = test_df["label"]
    X_test = test_df.drop(["label"], axis=1)
    X_test_1 = preprocess_and_select_features(X_test, scaler_1, indices_1)
    X_test_2 = preprocess_and_select_features(X_test, scaler_2, indices_2)

    evaluate_and_plot(model_1, X_test_1, y_test, "test model 1")
    evaluate_and_plot(model_2, X_test_2, y_test, "test model 2")
    # evaluate_and_plot(model_2, X_test, y_test, "test model 1")

    # 调用函数进行评估
    evaluate_and_plot_2(model_1, model_2, X_test_1, X_test_2, y_test, "Combined Model Evaluation")
    print(1)
    print(1)
