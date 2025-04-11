"""
数据预处理模块
功能：
1. 加载FER2013表情数据集
2. 对图像数据进行归一化处理
3. 对标签进行one-hot编码
"""
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_path='fer2013.csv'):
    """
    加载并预处理数据
    Args:
        data_path: 数据集路径（CSV格式）
    Returns:
        X: 预处理后的图像数据 (num_samples, 48, 48, 1)
        y: one-hot编码后的标签
    """
    # 读取CSV文件
    data = np.loadtxt(data_path, delimiter=',', dtype='str', skiprows=1)
    
    # 提取像素数据和标签
    pixels = data[:, 1].astype('float32') / 255.0  # 归一化
    labels = data[:, 0].astype('int32')
    
    # 重塑为图像格式 (48x48像素，单通道)
    X = np.array([np.fromstring(pixel, sep=' ').reshape(48, 48, 1) for pixel in pixels])
    
    # 对标签进行one-hot编码
    y = to_categorical(labels)
    
    return X, y


def load_image_data(train_dir, test_dir, batch_size=32, img_size=(48, 48)):
    """
    加载图像数据并进行实时增强
    Args:
        train_dir: 训练集目录路径
        test_dir: 测试集目录路径
        batch_size: 批次大小
        img_size: 图像尺寸
    Returns:
        train_generator: 训练数据生成器
        test_generator: 测试数据生成器
        class_indices: 类别标签映射字典
    """
    # 创建数据增强生成器
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # 从目录加载数据
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator, train_generator.class_indices