
---


# Planar Data Classification with a Neural Network

This repository contains a Python implementation of a neural network with one hidden layer for classifying planar data (non-linearly separable data). The project is designed to help beginners understand the basics of neural networks, including data generation, model building, training, evaluation, and visualization.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Exporting the Model](#exporting-the-model)
7. [Contributing](#contributing)
8. [License](#license)

---

## Introduction
Planar data classification is a classic problem in machine learning where the goal is to classify data points that are not linearly separable. In this project, we use a neural network with one hidden layer to solve this problem. The implementation is done in Python using TensorFlow/Keras for building the model and Matplotlib for visualization.

---

## Features
- **Data Generation**: Generate planar data using `make_moons` from `sklearn`.
- **Neural Network**: Build a neural network with one hidden layer using TensorFlow/Keras.
- **Training and Evaluation**: Train the model and evaluate its performance on a test set.
- **Visualization**: Visualize the decision boundary and training history.
- **Model Export**: Save the trained model in various formats (H5, SavedModel, TensorFlow.js, TensorFlow Lite).



## Installation
To run this project, you need to have Python installed along with the following libraries:
- NumPy
- Matplotlib
- TensorFlow
- Scikit-learn

You can install the required libraries using `pip`:

```bash
pip install numpy matplotlib tensorflow scikit-learn
```

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/planar-data-classification.git
   cd planar-data-classification
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook planar_data_classification.ipynb
   ```

3. Run the notebook cells sequentially to:
   - Generate and visualize planar data.
   - Build, train, and evaluate the neural network.
   - Visualize the decision boundary and training history.
   - Save the trained model.

---

## Results
### Data Visualization
![image](https://github.com/user-attachments/assets/cfd0eedf-53fb-4a84-b889-48a94c1b4157)


### Decision Boundary
![image](https://github.com/user-attachments/assets/57b15ebd-1b97-48a5-bd25-789a31572a98)


### Training History
![image](https://github.com/user-attachments/assets/fb51faab-f96a-47ec-86bf-bec5180f3dad)


### Model Performance
- **Test Accuracy**: 98.50%

---

## Exporting the Model
The trained model can be exported in multiple formats for different use cases:
1. **H5 Format**:
   ```python
   model.save('planar_classification_model.h5')
   ```

2. **TensorFlow SavedModel Format**:
   ```python
   model.save('saved_model')
   ```

3. **TensorFlow.js**:
   ```python
   import tensorflowjs as tfjs
   tfjs.converters.save_keras_model(model, 'tfjs_model')
   ```

4. **TensorFlow Lite**:
   ```python
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   with open('model.tflite', 'wb') as f:
       f.write(tflite_model)
   ```

---

## Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Special thanks to the TensorFlow and Scikit-learn teams for their amazing libraries.
- Inspired by Andrew Ng's deep learning course.

---

## Connect with Me
- [LinkedIn](https://linkedin.com/in/adityainamdar1)


---

Enjoy exploring the world of neural networks! 

