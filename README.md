# Automated Music Genre Classification with Deep Learning Techniques

![Deep Learning](https://img.shields.io/badge/deep%20learning-TensorFlow-brightgreen)
![Python](https://img.shields.io/badge/python-3.x-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains a deep learning project aimed at classifying music genres using Convolutional Neural Networks (CNNs). The model analyzes audio signals to predict the genre of a music track, leveraging advanced deep learning techniques and the GTZAN Music Genre Dataset.

## Project Overview

This project involves the following steps:
- **Data Preprocessing**: Converting audio signals into Mel-Frequency Cepstral Coefficients (MFCCs) and spectrograms.
- **Model Building**: Implementing a CNN architecture to classify music genres.
- **Training and Evaluation**: Optimizing the model and evaluating its performance to achieve high classification accuracy.

## Dataset

The dataset used in this project is the [GTZAN Music Genre Dataset](http://marsyas.info/downloads/datasets.html), which includes 1,000 audio tracks categorized into 10 different genres: Blues, Classical, Country, Disco, Hip hop, Jazz, Metal, Pop, Reggae, and Rock.

## Repository Structure

- **`Audio_Classification.ipynb`**: Jupyter Notebook with the full implementation, including data preprocessing, model training, and evaluation.
- **`data/`**: Directory for the dataset (Note: The dataset is not included in the repository due to size constraints).
- **`models/`**: Directory containing saved models and checkpoints.
- **`results/`**: Directory for storing results, including metrics and plots.

## Installation

### Requirements

To run this project, you need the following:

- Python 3.x
- Jupyter Notebook
- Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow
  - librosa

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/SaiDhanyaa/Automated-Music-Genre-Classification.git
    cd Automated-Music-Genre-Classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Download the GTZAN dataset from [here](http://marsyas.info/downloads/datasets.html) and place it in the `data/` directory.
2. Open the `Audio_Classification.ipynb` notebook in Jupyter Notebook.
3. Follow the notebook instructions to preprocess the data, train the model, and evaluate the results.

## Results

The CNN model achieved the following accuracies:
- **Training Accuracy**: 88.33%
- **Validation Accuracy**: 82.47%
- **Test Accuracy**: 82.44%

These results demonstrate the model's capability to generalize across different music genres in the dataset.

## References

1. **Dhanyapriya Somasundaram, Misha Seroukhov** - "Automated Music Genre Classification with Deep Learning Techniques," MIS 548, Eller College of Management, University of Arizona, Tucson, AZ. [PDF Link](./path/to/MIS_548_GROUP_4_REPORT.pdf)
2. **G. Tzanetakis and P. Cook** - "Musical genre classification of audio signals," IEEE Transactions on Speech and Audio Processing, vol. 10, no. 5, pp. 293-302, July 2002. [DOI](https://doi.org/10.1109/TSA.2002.800560)
3. **K. Zaman, M. Sah, C. Direkoglu and M. Unoki** - "A Survey of Audio Classification Using Deep Learning," IEEE Access, vol. 11, pp. 106620-106649, 2023. [DOI](https://doi.org/10.1109/ACCESS.2023.3318015)
4. **W. Zhu and M. Omar** - "Multiscale Audio Spectrogram Transformer for Efficient Audio Classification," ICASSP 2023. [DOI](https://doi.org/10.1109/ICASSP49357.2023.10096513)
5. **Lei Yang and Hongdong Zhao** - "Sound Classification Based on Multihead Attention and Support Vector Machine," Mathematical Problems in Engineering, vol. 2021. [DOI](https://doi.org/10.1155/2021/9937383)

## Contributing

Contributions are welcome! If you would like to contribute, please fork the repository and use a feature branch. Pull requests are gladly accepted.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- GTZAN Music Genre Dataset by G. Tzanetakis and P. Cook (2002).
- TensorFlow and Keras documentation for deep learning resources.

## Contact

For any questions or feedback, please contact [Dhanyapriya Somasundaram](mailto:dhanyapriyas@arizona.edu).
