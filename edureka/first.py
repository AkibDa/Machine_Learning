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
scatter_matrix(dataset)
plt.show()
