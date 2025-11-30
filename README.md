# CIFAR-10 Image Classification with CNN

This project implements a Convolutional Neural Network (CNN) using **TensorFlow** and **Keras** to classify images into 10 distinct categories. It utilizes the popular CIFAR-10 dataset and includes functionality to test the model on custom external images.

## 📋 Project Overview

* **Dataset:** CIFAR-10 (60,000 32x32 color images).
* **Classes:** Plane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
* **Model Architecture:** Convolutional Neural Network (CNN) with Max Pooling and Dense layers.
* **Performance:** Achieves ~65% accuracy on the validation set after 10 epochs.
* **Input Shape:** 32x32 pixels (RGB).

## 🛠️ Technologies Used

* **Python 3.x**
* **TensorFlow / Keras** (Model building and training)
* **OpenCV** (Image loading and processing)
* **Matplotlib** (Data visualization)
* **NumPy** (Array manipulation)

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ahmedwisam1/Image-classification-with-Neural-Networks.git](https://github.com/ahmedwisam1/Image-classification-with-Neural-Networks.git)
    cd YOUR_REPO_NAME
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy matplotlib opencv-python
    ```

3.  **Run the Notebook:**
    Open `Image_Classification_with_Neural_Networks.ipynb` in Jupyter Notebook or Google Colab and run all cells.

## 📸 Custom Prediction

The notebook allows you to test the model on local images:
1.  Ensure `image_classifier.keras` is saved (this happens automatically after training).
2.  Place an image (e.g., `plane.jpeg`) in the project directory.
3.  Update the filename in the final code cell if necessary:
    ```python
    img = cv.imread('your_image.jpeg')
    ```
4.  Run the prediction cell to see the classification result.
