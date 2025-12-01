# Handwritten-digit-recognizer

🧠 Handwritten Digit Recognizer using CNN

A deep learning project built by Omkar Lotake and Akash Shinde using Convolutional Neural Networks (CNN) to classify handwritten digits from the MNIST dataset with high accuracy.

📌 Project Overview

This project demonstrates how a CNN model can learn patterns from handwritten digit images (0–9) and accurately classify them.
We trained the model on the MNIST dataset and visualized predictions, accuracy, and loss graphs.

The goal was to understand computer vision fundamentals and build a complete digit recognition pipeline.

🚀 Features

🔢 Recognizes handwritten digits (0–9)

🧠 Built with Convolutional Neural Networks

📊 Training & validation accuracy visualization

🖼 Shows real predictions with image samples

⚙️ Clean and fully working Python code

🎯 Achieved ~99% accuracy on test data

🧰 Tech Stack

Python

TensorFlow / Keras

NumPy

Matplotlib

MNIST Dataset

📂 Project Structure
├── digit_recognizer.py    # Main program
├── README.md              # Project documentation
└── sample_outputs/        # Prediction images & graphs

🧪 How It Works

Load MNIST dataset

Normalize and reshape the images

Build a CNN model:

Conv2D → ReLU

MaxPooling

Conv2D → ReLU

MaxPooling

Flatten

Dense (128 neurons)

Dropout (0.5)

Dense (10 neurons with Softmax)

Train model for 5 epochs

Evaluate accuracy

Display predictions and graphs

📝 Code Used

The project includes the full source code:

Training CNN

Evaluating results

Testing predictions

Plotting accuracy and loss

📈 Model Performance

Test Accuracy: ~99%

Strong generalization & stable training

Correctly predicts most digits from the test set
<img width="1189" height="543" alt="download (1)" src="https://github.com/user-attachments/assets/a11b0caa-0377-4847-87e5-c41bf4163065" />
