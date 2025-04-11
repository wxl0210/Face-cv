"""
模型训练模块
功能：
1. 加载预处理数据
2. 初始化并编译模型
3. 配置训练参数和回调函数
4. 保存训练好的模型
"""
from data_loader import load_image_data
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model():
    # 加载图像数据
    train_gen, test_gen, class_map = load_image_data(
        train_dir='train',
        test_dir='test',
        batch_size=64
    )

    # 构建模型
    model = build_model(input_shape=(48,48,1), num_classes=len(class_map))
    model.summary()

    # 配置回调函数
    callbacks = [
        ModelCheckpoint('best_emotion_model.keras', save_best_only=True),
        EarlyStopping(patience=15, restore_best_weights=True)
    ]

    # 开始训练
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        validation_data=test_gen,
        validation_steps=test_gen.samples // test_gen.batch_size,
        epochs=20,
        callbacks=callbacks
    )

    model.save('emotion_recognition_v2.keras', save_format='keras')
    return history

if __name__ == '__main__':
    train_model()