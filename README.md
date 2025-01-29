<img width="100" align="left" style="float: left; margin: 0 15px 0 0;" alt="Neural Network" src="https://raw.githubusercontent.com/TecnoSasso/SimpleNeuralNetwork/main/img/logo.png">
# Simple Neural Network from Scratch

This project implements a simple but fully connected neural network from scratch in Python, without using deep learning libraries such as TensorFlow or PyTorch. The network is designed to be trained on the MNIST dataset and can be used directly from the terminal.

## Features
- Implements a multi-layer perceptron (MLP) with customizable architecture.
- Uses ReLU (Rectified Linear Unit) as the activation function.
- Supports training and evaluation on the MNIST dataset.
- Can process custom 28x28 grayscale images for digit classification.
- Includes save and load functionality for trained models.

## Requirements
Make sure you have the following dependencies installed:

```bash
pip install pillow
```

## Installation
To set up the project, you have two options:

1. **Download the MNIST Dataset Files Manually:**
   - Obtain the `mnist_train.csv` and `mnist_test.csv` files from a reliable source.
   - Place these files in the project's root directory.

2. **Extract the Provided `project.zip` File:**
   - Download the `project.zip` file from the repository's [Releases](https://github.com/TecnoSasso/NeuralNetwork/releases) section.
   - Extract its contents to your desired location.

After setting up the dataset, ensure you have the required dependencies installed:

```bash
pip install pillow
```

Now, you can proceed to run the project as described in the Usage section.

## Usage

### Training the Network
Run the script and select the training mode:

```bash
python Neural-Network.py
```
Then, choose `t` for training and specify the batch size and learning rate when prompted.


<img alt="Train" src="https://raw.githubusercontent.com/TecnoSasso/SimpleNeuralNetwork/main/img/average_cost.png">

### Evaluating Accuracy
After training, you can evaluate the network's accuracy on the test dataset by selecting `a`.


<img alt="Accuracy Example" src="https://raw.githubusercontent.com/TecnoSasso/SimpleNeuralNetwork/main/img/accuracy.png">

### Predicting from an Image
To classify a custom image, use the `i` option and provide the image filename (must be 28x28 grayscale).

### Saving and Loading Models
- Save the current model with `s`. Ensure the `./networkData/` directory exists.
- Load a previously saved model with `l`.

## File Structure
- `mnist_train.csv` - Training dataset
- `mnist_test.csv` - Test dataset
- `networkData/weights.txt` - Saved weights
- `networkData/biases.txt` - Saved biases

  
<img alt="File Structure" src="https://raw.githubusercontent.com/TecnoSasso/SimpleNeuralNetwork/main/img/File_Structure.png">

## Notes
- The network uses backpropagation with gradient descent for training.
- Ensure the MNIST dataset files are present in the working directory.

## Author
Developed by **SASSO**.
