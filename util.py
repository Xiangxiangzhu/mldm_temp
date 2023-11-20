import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


def evaluate_and_plot(model, x_test_, y_test_, title):
    # 预测并打印分类报告
    y_pred_test = model.predict(x_test_)
    print(f"Classification Report for {title}:")
    print(classification_report(y_test_, y_pred_test))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_, y_pred_test))

    # 绘制ROC曲线
    y_prob_test = model.predict_proba(x_test_)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test_, y_prob_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.figure()
    plt.plot(fpr_test, tpr_test, label=f'ROC curve for {title} (area = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {title}')
    plt.legend(loc="lower right")
    plt.show()
