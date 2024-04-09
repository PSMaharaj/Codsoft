import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("Titanic-Dataset.csv")
print(df)
target = df.Survived
print(target)
df = df.drop(["Name", "Ticket", "Cabin", "Survived"], axis=1)
df.isna().sum()
print("After Droping : \n",df)
age_median = df.Age.median()
print(age_median)
df.Age = df.Age.fillna(math.floor(age_median))
print(df.isna().sum())
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder',
                                      OneHotEncoder(handle_unknown='ignore'),
                                      ['Sex', 'Embarked'])
                                    ],
                       remainder='passthrough',
                       sparse_threshold=0)
data_encoded = ct.fit_transform(df)
new_df = pd.DataFrame(data_encoded, columns=ct.get_feature_names_out())
new_df = new_df.drop("encoder__Embarked_nan", axis=1)
print(new_df)
x_train,x_test,y_train,y_test = train_test_split(new_df, target, test_size=0.2)


