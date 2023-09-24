# Breast Cancer Classification using Logistic Regression

## Introduction
This Python script is designed to perform breast cancer classification using Logistic Regression. It utilizes the scikit-learn library for machine learning tasks.

## Prerequisites
Before running the script, make sure you have the following prerequisites installed:

- Python (3.x recommended)
- scikit-learn
- pandas
- numpy

You can install these dependencies using `pip`:

```bash
pip install scikit-learn pandas numpy
```

## Usage
1. Clone or download this repository to your local machine.

2. Place the dataset file `breast_cancer.csv` in the same directory as the script.

3. Open a terminal or command prompt and navigate to the directory where the script is located.

4. Run the script using the following command:

```bash
python logisticreg.py
```

## Description
- The script loads the breast cancer dataset from the `breast_cancer.csv` file.

- It splits the dataset into training and test sets using a 80-20 split.

- A Logistic Regression classifier is trained on the training data.

- The script then predicts the labels for the test set.

- It calculates and displays the confusion matrix and accuracy of the classifier on the test set.

- Cross-validation is performed using 10-fold cross-validation, and the accuracy along with the standard deviation of the accuracy scores is displayed.

## Results
- The confusion matrix shows the true positive, true negative, false positive, and false negative values.

- The accuracy score is calculated and displayed as a percentage.

- The script also calculates the mean accuracy and standard deviation using cross-validation to assess the model's performance on different subsets of the training data.

## License
This code is provided under the [MIT License](LICENSE).

Feel free to modify and use the code for your own projects. If you find it helpful, please consider giving it a star!

For any questions or issues, please open an [issue](https://github.com/SVatghub/Breast-Cancer-detection-using-Logistic-Regression/issues) in the repository.