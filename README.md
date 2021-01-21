# Task-6-Prediction-using-Decision-Tree-Algorithm


BAMMIDI PREM KUMAR
GRIP THE SPARKS FOUNDATION
TASK6: Prediction using Decision Tree Algorithm - Create the Decision Tree classifier and visualize it graphically
In [2]:
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
In [4]:
dataset = pd.read_csv("Iris.csv")
In [5]:
dataset.head()
Out[5]:
Id	SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
0	1	5.1	3.5	1.4	0.2	Iris-setosa
1	2	4.9	3.0	1.4	0.2	Iris-setosa
2	3	4.7	3.2	1.3	0.2	Iris-setosa
3	4	4.6	3.1	1.5	0.2	Iris-setosa
4	5	5.0	3.6	1.4	0.2	Iris-setosa
In [6]:
dataset = dataset.drop(['Id'],axis = 1)
dataset.head()
Out[6]:
SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
0	5.1	3.5	1.4	0.2	Iris-setosa
1	4.9	3.0	1.4	0.2	Iris-setosa
2	4.7	3.2	1.3	0.2	Iris-setosa
3	4.6	3.1	1.5	0.2	Iris-setosa
4	5.0	3.6	1.4	0.2	Iris-setosa
In [7]:
dataset.isnull().sum()
Out[7]:
SepalLengthCm    0
SepalWidthCm     0
PetalLengthCm    0
PetalWidthCm     0
Species          0
dtype: int64
In [8]:
sns.pairplot(dataset)
Out[8]:
<seaborn.axisgrid.PairGrid at 0x1c3baf0a610>

In [11]:
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
In [13]:
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
y = lab.fit_transform(y)
y
Out[13]:
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
In [14]:
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(x,y)
In [15]:
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
Out[15]:
DecisionTreeClassifier()
In [16]:
dt.score(X_train,y_train)
Out[16]:
1.0
In [18]:
y_pred = dt.predict(x_test)
y_pred
Out[18]:
array([0, 0, 1, 0, 1, 2, 1, 0, 1, 2, 0, 2, 0, 2, 2, 0, 1, 1, 2, 1, 2, 2,
       1, 0, 2, 1, 1, 1, 2, 1, 2, 0, 2, 2, 2, 1, 0, 2])
In [20]:
df = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df.head()
Out[20]:
Actual	Predicted
0	0	0
1	0	0
2	1	1
3	0	0
4	1	1
In [24]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
[[10  0  0]
 [ 0 13  0]
 [ 0  0 15]]
Out[24]:
1.0
In [ ]:
