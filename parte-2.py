from sklearn import preprocessing
import time
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import display 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def calc_knn(nombre1, nombre2):
  my_data = pd.read_csv(nombre1, delimiter=",")
  X_train = my_data[['accessiblity','cs_research','designer','entertainment','expert_user','fanatic','gamer','old_hardware','privacy','programmer','reliable','security','software','tech_support','willing_pay']].values

  my_data_2 = pd.read_csv(nombre2, delimiter=",")
  X_test = my_data_2[['accessiblity','cs_research','designer','entertainment','expert_user','fanatic','gamer','old_hardware','privacy','programmer','reliable','security','software','tech_support','willing_pay']].values


  le_accessiblity= preprocessing.LabelEncoder()
  le_accessiblity.fit(['Si','No'])
  X_train[:,0] = le_accessiblity.transform(X_train[:,0])
  X_test[:,0] = le_accessiblity.transform(X_test[:,0])

  le_cs_research= preprocessing.LabelEncoder()
  le_cs_research.fit(['Si','No'])
  X_train[:,1] = le_cs_research.transform(X_train[:,1])
  X_test[:,1] = le_cs_research.transform(X_test[:,1])

  le_designer= preprocessing.LabelEncoder()
  le_designer.fit(['Si','No'])
  X_train[:,2] = le_designer.transform(X_train[:,2])
  X_test[:,2] = le_designer.transform(X_test[:,2])

  le_entertainment= preprocessing.LabelEncoder()
  le_entertainment.fit(['Si','No'])
  X_train[:,3] = le_entertainment.transform(X_train[:,3])
  X_test[:,3] = le_entertainment.transform(X_test[:,3])

  le_expert_user= preprocessing.LabelEncoder()
  le_expert_user.fit(['Si','No'])
  X_train[:,4] = le_expert_user.transform(X_train[:,4])
  X_test[:,4] = le_expert_user.transform(X_test[:,4])

  le_fanatic= preprocessing.LabelEncoder()
  le_fanatic.fit(['Si','No'])
  X_train[:,5] = le_fanatic.transform(X_train[:,5])
  X_test[:,5] = le_fanatic.transform(X_test[:,5])

  le_gamer= preprocessing.LabelEncoder()
  le_gamer.fit(['Si','No'])
  X_train[:,6] = le_gamer.transform(X_train[:,6])
  X_test[:,6] = le_gamer.transform(X_test[:,6])

  le_old_hardware= preprocessing.LabelEncoder()
  le_old_hardware.fit(['Si','No'])
  X_train[:,7] = le_old_hardware.transform(X_train[:,7])
  X_test[:,7] = le_old_hardware.transform(X_test[:,7])

  le_privacy = preprocessing.LabelEncoder()
  le_privacy.fit(['Si','No'])
  X_train[:,8] = le_privacy.transform(X_train[:,8])
  X_test[:,8] = le_privacy.transform(X_test[:,8])

  le_programmer= preprocessing.LabelEncoder()
  le_programmer.fit(['Si','No'])
  X_train[:,9] = le_programmer.transform(X_train[:,9])
  X_test[:,9] = le_programmer.transform(X_test[:,9])

  le_reliable= preprocessing.LabelEncoder()
  le_reliable.fit(['Si','No'])
  X_train[:,10] = le_reliable.transform(X_train[:,10])
  X_test[:,10] = le_reliable.transform(X_test[:,10])

  le_security= preprocessing.LabelEncoder()
  le_security.fit(['Si','No'])
  X_train[:,11] = le_security.transform(X_train[:,11])
  X_test[:,11] = le_security.transform(X_test[:,11])

  le_software= preprocessing.LabelEncoder()
  le_software.fit(['Si','No'])
  X_train[:,12] = le_software.transform(X_train[:,12])
  X_test[:,12] = le_software.transform(X_test[:,12])

  le_tech_support= preprocessing.LabelEncoder()
  le_tech_support.fit(['Si','No'])
  X_train[:,13] = le_tech_support.transform(X_train[:,13])
  X_test[:,13] = le_tech_support.transform(X_test[:,13])

  le_willing_pay= preprocessing.LabelEncoder()
  le_willing_pay.fit(['Si','No'])
  X_train[:,14] = le_willing_pay.transform(X_train[:,14])
  X_test[:,14] = le_willing_pay.transform(X_test[:,14])


  lb = LabelBinarizer()
  y_train = my_data['class']
  y_train_b = lb.fit_transform(y_train.to_numpy())

  y_test = my_data_2['class']
  y_test_b = lb.fit_transform(y_test.to_numpy())


  neighbors = [1, 3, 5, 7, 9, 11, 13, 15]
  test_accuracy = np.empty(len(neighbors))
  test_precision = np.empty(len(neighbors))
  test_recall = np.empty(len(neighbors))
  test_f1  = np.empty(len(neighbors))
  test_time  = np.empty(len(neighbors))

  for i,k in enumerate(neighbors):
      knn = KNeighborsClassifier(n_neighbors=k)
      
      knn.fit(X_train, y_train_b.reshape(-1))
      start_time = time.time()
      y_pred = knn.predict(X_test)
      elapsed_time = time.time()-start_time
      test_time[i] = elapsed_time

      test_accuracy[i] = accuracy_score(y_test_b, y_pred) 
      test_precision[i] = precision_score(y_test_b,y_pred)
      test_recall[i] = recall_score(y_test_b, y_pred)
      test_f1[i] = f1_score(y_test_b, y_pred)

  numpy_data = np.array([test_accuracy,test_precision,test_recall,test_f1, test_time])
  df = pd.DataFrame(data=numpy_data, index=["Accuracy","Precision","Recall", "F1","Predict training time"], columns=neighbors)
  df.to_csv(path_or_buf="Parte-2.csv", index=True)
  display(df)



def main():
  nombre = input("Ingrese el nombre del archivo de entrenamiento: ")
  nombre2 = input("Ingrese el nombre del archivo de pruebas: ")
  calc_knn(nombre, nombre2)
main()
