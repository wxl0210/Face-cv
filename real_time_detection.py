import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 加载预训练的表情识别模型
model = load_model('emotion_recognition_v2.keras')

# 定义表情类别标签（需要与训练时的class_indices保持一致）
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# 初始化Haar级联分类器（需要haarcascade_frontalface_default.xml文件）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 捕获视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图像（Haar特征需要灰度输入）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用Haar特征检测人脸
    """
    Haar级联工作原理：
    1. 使用积分图快速计算矩形特征
    2. 通过Adaboost算法选择重要特征
    3. 级联分类器逐步过滤非人脸区域
    """
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )

    for (x, y, w, h) in faces:
        # 提取人脸ROI区域
        face_roi = gray[y:y+h, x:x+w]
        
        # 预处理（与训练数据一致）
        resized = cv2.resize(face_roi, (48, 48))
        normalized = resized / 255.0
        input_data = np.expand_dims(np.expand_dims(normalized, -1), 0)

        # 表情预测
        predictions = model.predict(input_data)
        emotion_label = EMOTION_LABELS[np.argmax(predictions)]
        confidence = np.max(predictions)

        # 绘制检测框和标签
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_text = f'{emotion_label} ({confidence:.2f})'
        cv2.putText(frame, label_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Real-time Emotion Detection', frame)
    
    # 退出条件
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()