import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
iris_df = pd.read_csv('IRIS.csv')
print(iris_df.info())
print(iris_df.shape)
print(iris_df.describe())
print(iris_df.isna().sum())
sns.histplot(data=iris_df,x='sepal_length')
plt.title("The distribution of sepal_length")
plt.show()
sns.histplot(data=iris_df,x='sepal_width')
plt.title("The distribution of sepal_width ")
plt.show()
sns.histplot(data=iris_df,x='petal_length')
plt.title("The distribution of petal_length")
plt.show()
sns.histplot(data=iris_df,x='petal_width')
plt.title("The distribution of petal_width")
plt.show()
sns.pairplot(iris_df,hue='species')
plt.show()
encoder = LabelEncoder()
iris_df['species'] = encoder.fit_transform(iris_df['species'])
iris_df.head()
