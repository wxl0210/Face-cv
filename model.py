"""
卷积神经网络模型定义
架构：
1. 输入层 (48x48x1)
2. 3个卷积块（Conv2D + BatchNorm + MaxPooling2D）
3. 全连接分类层
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization

def build_model(input_shape=(48,48,1), num_classes=7):
    """
    构建CNN模型
    Args:
        input_shape: 输入张量形状 (高度, 宽度, 通道数)
        num_classes: 分类类别数
    Returns:
        编译好的Keras模型
    """
    model = Sequential([
        # 第一个卷积块
        Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # 第二个卷积块
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        # 第三个卷积块
        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.35),

        # 分类层
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model