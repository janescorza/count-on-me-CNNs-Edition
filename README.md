# Count On Me: CNNs Edition 🎓🌟
**"Count On Me: CNNs Edition"** propels your neural network journey forward, diving into the exciting world of Convolutional Neural Networks (CNNs). This educational adventure uses hand sign recognition to teach not only the basics of number recognition but also the intricacies of CNNs, making learning both fun and technically enriching.

## Features 🚀
- **Custom CNN from Scratch**: Dive into the nuts and bolts of building and training a CNN entirely from scratch. Watch as it learns and evolves with each epoch! Dive into the code to see who both the forward and backward models are built in detail.
- **TensorFlow Model Mode**: Utilize TensorFlow to streamline model building and training, featuring high accuracy and efficiency.
- **ResNet50 Model Mode**: Explore the capabilities of advanced deep learning with the ResNet50 mode. This mode employs a renowned architecture for its powerful feature extraction capabilities in image recognition tasks. 


## Getting Started 🌟
Jump into this hands-on neural network experience with a few simple steps:

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/janescorza/count-on-me-CNNs-Edition.git
   ```
2. **Create a virtual environment**: 

Run `python -m venv env` and then `source env/bin/activate` to create and activate the virtual environment

3. **Install Requirements**: 
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch the Application**: 
   ```bash
   python main.py
   ```
   and follow the on-screen prompts to choose a mode and interact with the models.

## Prerequisites 📋
Ensure you have Python 3.6 or newer. Currently, the custom CNN does not utilize GPU acceleration and is optimized for educational purposes rather than performance.

## Modes of Operation 🔄
### Mode 1: Custom CNN from Scratch
Experience the raw mechanics of CNNs by training a model from the ground up. Adjust parameters and observe the effects in real-time:
- **Training and Learning**: Despite not being the fastest due to its lack of GPU optimization, the model shows consistent learning across epochs.
- **Educational Focus**: Designed for demonstration, this mode helps you understand the fundamentals of CNNs.
- **Future Enhancements**: Experiment with adding more layers, introducing regularization, or employing data augmentation to see how they can boost the model's performance.


### Mode 2: TensorFlow Model
Experience high efficiency and quick training times with the TensorFlow implementation of the convolutional network. This mode is optimized to leverage TensorFlow's capabilities, enabling it to consistently achieve over 90% accuracy on the test set during my trials:
- **High Accuracy**: Achieves excellent results on the predefined test set due to the optimized network architecture.
- **Speed and Efficiency**: Capable of being trained on demand for any number of epochs due to its efficient use of hardware.
- **Future Enhancements**: While the model performs admirably on the test set, its effectiveness on real-world, diverse data could be enhanced by expanding the dataset to include more varied examples from different environments.

### Mode 3: ResNet50 Model
Dive into advanced deep learning with the ResNet50 model mode, leveraging a pre-trained network structure renowned for its powerful feature extraction capabilities:
- **State-of-the-Art Architecture**: Utilize ResNet50, a model that has excelled in image recognition tasks, and that you will train to grasp complex patterns in the hand signs.
- **Hands-On Learning with Advanced Models**: Offers a hands-on approach to understanding how more complex architectures are built with simpler building blocks.
- **Building on Previous Knowledge**: By engaging with earlier modes, readers like you will have a foundational understanding of neural networks, enabling you to appreciate and grasp the complexities of the ResNet50 architecture more effectively.



## Dataset and Neural Network Considerations 🧠
The CNN model is trained for educational purposes on a relatively small dataset. It provides a hands-on way to observe and understand model behavior and training dynamics:
- **Non-Optimized Performance**: The current setup is not designed for high-speed GPU execution.
- **Potential for Deeper Learning**: Extending the training to more epochs or enhancing the model architecture could yield more significant insights and performance improvements.

## How It Works 🔍
From initializing parameters to processing data and training the model, "Count On Me: CNNs Edition" invites you on an insightful journey into the world of deep learning. This project builds on the foundational knowledge from the previous "Count On Me" project, advancing to more complex neural network structures.

## Contributing 🤝
Your insights and contributions are welcome! Feel free to fork the repository, push your enhancements, or share your thoughts on further improvements.