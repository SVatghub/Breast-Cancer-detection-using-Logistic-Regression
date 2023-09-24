from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
df = pd.read_csv('breast_cancer.csv')

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# training
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# print(np.concatenate((y_pred.reshape(len(y_pred), 1),
#       y_test.reshape(len(y_test), 1)), axis=1))

# getting the accuracy
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("confusion matrix :\n {}".format(cm))
print("Accuracy : {:.2f}".format(accuracy*100))

# getting the cross validation score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy using cross val score : {:.2f}".format(accuracies.mean()*100))
print("Standard Deviation using cross val score : {:.2f}".format(
    accuracies.std()*100))
