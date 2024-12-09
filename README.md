Emotion Recognition with Convolutional Neural Networks
This project explores the fascinating world of emotion recognition using Convolutional Neural Networks (CNNs) in Python. The goal was to create a model capable of classifying human facial expressions into seven categories: surprise, fear, angry, neutral, sad, disgust, and happy.

How It Works
Dataset: Used the FER-2013 dataset, containing 27,000 48x48 grayscale images of facial expressions.
Tools: PyTorch, NumPy, OpenCV, and Kaggle.
Model: Built a CNN that processes pixel values (normalized between 0 and 1) and predicts the emotion category.
Training: Ran the model for up to 300 epochs with preprocessing and optimization techniques to improve performance.

Issues Faced:
Trying to implement a method so that the dataset is auto downloaded into the computer proved to be ineffective for the given dataset (FER 2013). Switched to a standard approach and have acceptable results.

Results
After multiple iterations:
Achieved a final accuracy of 55.42% on the test dataset.
Improved performance by increasing model complexity, adding layers, and refining preprocessing techniques.
Despite the subjective and ambiguous nature of the dataset, the model significantly outperformed random guessing (14%) and even human attempts (30%).

While the model isn't perfect, it demonstrates the potential of CNNs for emotion recognition and serves as a stepping stone for future improvements.
