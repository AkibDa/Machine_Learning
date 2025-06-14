import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
name = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
dataset = pandas.read_csv(url, names=name)

# print(dataset.shape)
# print(dataset.head(30))
# print(dataset.describe())
# print(dataset.groupby('species').size())

# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()
# dataset.hist()
# plt.show()
# scatter_matrix(dataset)
# plt.show()

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 6
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

seed = 6
scoring = 'accuracy'

# Spot Check Algorithm
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
  kfold = model_selection.KFold(n_splits=10)
  cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)