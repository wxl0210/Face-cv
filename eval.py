"""
模型验证与可视化模块
功能：
1. 加载训练历史记录和测试数据
2. 绘制训练过程准确率/损失曲线
3. 生成混淆矩阵和分类报告
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def visualize_training(history):
    """
    可视化训练过程中的指标变化
    Args:
        history: 训练历史对象（model.fit返回值）
    """
    plt.figure(figsize=(12, 5))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能并生成分类报告
    Args:
        model: 训练好的Keras模型
        X_test: 测试集图像数据
        y_test: 测试集真实标签
    """
    # 模型预测
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 生成分类报告
    print('\n分类报告：')
    print(classification_report(y_true, y_pred_classes))

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == '__main__':
    from tensorflow.keras.models import load_model
    from data_loader import load_data
    
    # 加载模型和测试数据
    model = load_model('emotion_cnn.h5')
    X, y = load_data()
    
    # 划分测试集（使用最后20%数据）
    test_start = int(0.8 * len(X))
    X_test, y_test = X[test_start:], y[test_start:]
    
    # 执行评估
    evaluate_model(model, X_test, y_test)