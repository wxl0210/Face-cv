本系统采用计算机视觉与深度学习技术实现实时面部表情识别，技术架构包含：
- 前端采集：OpenCV视频流处理
- 特征提取：Haar级联人脸检测
- 核心模型：基于FER2013训练的卷积神经网络
- 预测输出：7种基本情绪分类（Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise）# Face-cv
下载FER2013数据集（https://www.kaggle.com/datasets/msambare/fer2013）
