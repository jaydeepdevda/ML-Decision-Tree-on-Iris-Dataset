from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

# load data set from packages
data_set = datasets.load_iris()

# assign features and its class label
X = data_set.data
y = data_set.target

print("== Metadata ==")
# size of data set
print("Total Data Rows", X.shape[0])
print("No of Features",X.shape[1])
print("Class Labels",set(y))

# train and test data set splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

print("Train Data Rows", len(X_train))
print("Test Data Rows", len(X_test))

# decision tree classification model
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X_train, y_train)

# predict over testing data
pred = clf.predict(X_test)
accuracy = accuracy_score(pred ,y_test)

# print accuracy
print("Accuracy", accuracy)

print("Input Sample",X_test[0], y_test[0])
# sample one data set # ouput 1
sample = [[6.3, 2.3, 4.4, 1.3]]
op = clf.predict(sample)
print("Predicted class", op)
