from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


data_set = "tennis.csv"

df  = pd.read_csv(data_set)

lb = LabelEncoder()
df['outlook'] = lb.fit_transform(df['outlook']) 
df['temp'] = lb.fit_transform(df['temp'] ) 
df['humidity'] = lb.fit_transform(df['humidity'] ) 
df['windy'] = lb.fit_transform(df['windy'] )   
df['play'] = lb.fit_transform(df['play'] ) 

X = df.drop(['play'], axis = 1)
y = df['play']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
accuracy = accuracy_score(pred,y_test)

print(accuracy)

opfile = open("dtree.dot", 'w')
export_graphviz(clf, out_file = opfile, feature_names = X.columns)
opfile.close()

