from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QFileDialog
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow
import keras
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Ui_Parkinsons_Diseases(object):
    def setupUi(self, Parkinsons_Diseases):
        Parkinsons_Diseases.setObjectName("Parkinsons_Diseases")
        Parkinsons_Diseases.resize(1300,900)
        Parkinsons_Diseases.setStyleSheet("background-color:white;\n"
"border-style: outset;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"border-color:black;")
        self.centralwidget = QtWidgets.QWidget(Parkinsons_Diseases)
        self.centralwidget.setObjectName("centralwidget")
        self.Browse_file = QtWidgets.QPushButton(self.centralwidget)
        self.Browse_file.setGeometry(QtCore.QRect(480, 90, 161, 51))
        self.Browse_file.setStyleSheet("background-color:blue;\n"
"color: white;\n"
"border-style: outset;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"border-color:black;\n"
"font:bold 14px;\n"
"padding :6px;\n"
"min-width:10px;")
        self.Browse_file.setObjectName("Browse_file")
        self.comboBox1 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox1.setGeometry(QtCore.QRect(690, 200, 331, 51))
        self.comboBox1.setStyleSheet("background-color:lightGreen;\n"
"color: black;\n"
"border-style: outset;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"border-color:black;\n"
"font: 20px;\n"
"padding :6px;\n"
"min-width:10px;")
        self.comboBox1.setObjectName("comboBox1")
        self.comboBox1.addItem("")
        self.comboBox1.addItem("")
        self.comboBox1.addItem("")
        self.comboBox1.addItem("")
        self.comboBox1.addItem("")
        self.comboBox1.addItem("")
        self.comboBox1.addItem("")
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(690, 150, 331, 51))
        self.label_1.setStyleSheet("background-color:white;\n"
"color: black;\n"
"border-style: outset;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"border-color:black;\n"
"font:25px;\n"
"padding :6px;\n"
"min-width:10px;")
        self.label_1.setObjectName("label_1")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(170, 160, 291, 71))
        self.label_2.setStyleSheet("background-color:yellow;\n"
"color: black;\n"
"border-style: outset;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"border-color:black;\n"
"font:bold 20px;\n"
"min-width:10px;")
        self.label_2.setObjectName("label_2")
        self.Accuracy1 = QtWidgets.QLabel(self.centralwidget)
        self.Accuracy1.setGeometry(QtCore.QRect(80, 260, 491, 51))
        self.Accuracy1.setStyleSheet("backgroud-color:white;\n"
"font:bold 18px")
        self.Accuracy1.setObjectName("Accuracy1")
        self.Accuracy2 = QtWidgets.QLabel(self.centralwidget)
        self.Accuracy2.setGeometry(QtCore.QRect(80, 320, 491, 51))
        self.Accuracy2.setStyleSheet("backgroud-color:white;\n"
"font:bold 18px;")
        self.Accuracy2.setObjectName("Accuracy2")
        self.Accuracy3 = QtWidgets.QLabel(self.centralwidget)
        self.Accuracy3.setGeometry(QtCore.QRect(80, 380, 491, 51))
        self.Accuracy3.setStyleSheet("backgroud-color:white;\n"
"font:bold 18px")
        self.Accuracy3.setObjectName("Accuracy3")
        self.Accuracy4 = QtWidgets.QLabel(self.centralwidget)
        self.Accuracy4.setGeometry(QtCore.QRect(80, 440, 491, 51))
        self.Accuracy4.setStyleSheet("backgroud-color:white;\n"
"font:bold 18px")
        self.Accuracy4.setObjectName("Accuracy4")

        self.Accuracy5 = QtWidgets.QLabel(self.centralwidget)
        self.Accuracy5.setGeometry(QtCore.QRect(80, 500, 491, 51))
        self.Accuracy5.setStyleSheet("backgroud-color:white;\n"
"font:bold 18px")
        self.Accuracy5.setObjectName("Accuracy5")

        self.Accuracy6 = QtWidgets.QLabel(self.centralwidget)
        self.Accuracy6.setGeometry(QtCore.QRect(80, 560, 491, 51))
        self.Accuracy6.setStyleSheet("backgroud-color:white;\n"
"font:bold 18px")
        self.Accuracy6.setObjectName("Accuracy6")

        self.Accuracy7 = QtWidgets.QLabel(self.centralwidget)
        self.Accuracy7.setGeometry(QtCore.QRect(80, 620, 491, 51))
        self.Accuracy7.setStyleSheet("backgroud-color:white;\n"
"font:bold 18px")
        self.Accuracy7.setObjectName("Accuracy7")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(580, 250,681 ,581 ))
        self.label.setText("")
        self.label.setTextFormat(QtCore.Qt.AutoText)
        #Adding label photo
        self.label.setPixmap(QtGui.QPixmap(""))
        self.label.setScaledContents(True)
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(850, 830, 141, 41))
        self.pushButton.setStyleSheet("background-color:skyblue;\n"
"color: black;\n"
"border-style: outset;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"border-color:black;\n"
"font:bold 14px;\n"
"\n"
"min-width:10px;")
        self.pushButton.setObjectName("pushButton")
        Parkinsons_Diseases.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Parkinsons_Diseases)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1145, 30))
        self.menubar.setObjectName("menubar")
        Parkinsons_Diseases.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Parkinsons_Diseases)
        self.statusbar.setObjectName("statusbar")
        Parkinsons_Diseases.setStatusBar(self.statusbar)

        self.retranslateUi(Parkinsons_Diseases)
        QtCore.QMetaObject.connectSlotsByName(Parkinsons_Diseases)

    def retranslateUi(self, Parkinsons_Diseases):
        _translate = QtCore.QCoreApplication.translate
        Parkinsons_Diseases.setWindowTitle(_translate("Parkinsons_Diseases", "Parkinson's Disease Projet"))
        self.Browse_file.setText(_translate("Parkinsons_Diseases", "Browse"))
        self.comboBox1.setItemText(0, _translate("Parkinsons_Diseases", "Linear Regression"))
        self.comboBox1.setItemText(1, _translate("Parkinsons_Diseases", "Logistic Regression"))
        self.comboBox1.setItemText(2, _translate("Parkinsons_Diseases", "Desicion Tree"))
        self.comboBox1.setItemText(3, _translate("Parkinsons_Diseases", "Support Vector Machines"))
        self.comboBox1.setItemText(4, _translate("Parkinsons_Diseases", "Random Forest"))
        self.comboBox1.setItemText(5, _translate("Parkinsons_Diseases", "XGBClassifier"))
        self.comboBox1.setItemText(6, _translate("Parkinsons_Diseases", "Neural Network"))
        self.label_1.setText(_translate("Parkinsons_Diseases", "Heat Map"))
        self.label_2.setText(_translate("Parkinsons_Diseases", "Accuracy Of the  All Models"))
        self.Accuracy1.setText(_translate("Parkinsons_Diseases", "Acurracy of Linear Regression : NuLL"))
        self.Accuracy2.setText(_translate("Parkinsons_Diseases", "Acurracy of Logistic Regression : NuLL"))
        self.Accuracy3.setText(_translate("Parkinsons_Diseases", "Acurracy of Desicion Tree : NuLL"))
        self.Accuracy4.setText(_translate("Parkinsons_Diseases", "Acurracy of Support Vector Machines : NuLL"))
        self.Accuracy5.setText(_translate("Parkinsons_Diseases", "Acurracy of Random Forest : NuLL"))
        self.Accuracy6.setText(_translate("Parkinsons_Diseases", "Acurracy of XGBClassifier : NuLL"))
        self.Accuracy7.setText(_translate("Parkinsons_Diseases", "Acurracy of Neural Network : NuLL"))
        self.pushButton.setText(_translate("Parkinsons_Diseases", "Submit"))
        self.Browse_file.clicked.connect(self.pushButton_handler)

    def pushButton_handler(self):
        self.open_dialog_box()
        
    def open_dialog_box(self):
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        print(path)
        self.Project(path)

    def Project(self,path):
        print(path)
        df = pd.read_csv(path)
        df=df.drop(['name'],axis=1)
        x = df.drop(['status'],axis=1)
        stdscaler = StandardScaler()
        x = np.array(stdscaler.fit_transform(x))
        y = df['status']
        X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=10)
        X_train=np.array(X_train)
        X_test=np.array(X_test)
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)
        #Linear Regression
        from sklearn.linear_model import LinearRegression
        model1 = LinearRegression()
        model1.fit(X_train,Y_train)
        Y_predmod1=model1.predict(X_test)
        for i,j in enumerate(Y_predmod1):
                if(j<0.5):
                        Y_predmod1[i]=0
                else:
                        Y_predmod1[i]=1
        x1=model1.score(X_test,Y_test)
        self.Accuracy1.setText("Acurracy of Linear Regression : {:.2f}".format(x1))
        #Confusion_matrix1
        cm=confusion_matrix(Y_test,Y_predmod1)
        plt.figure(figsize=(5,5))
        plt.title("Linear Regression")
        fg=sn.heatmap(cm,annot=True,cmap='Blues')
        figure = fg.get_figure()
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        figure.savefig('Linear_Regression.jpg',dpi=400)
       # plt.imshow(plt.imread("Linear_regression.jpg"))

        #Logistic Regression
        from sklearn.linear_model import LogisticRegression
        model2 = LogisticRegression()
        model2.fit(X_train,Y_train)
        Y_predmod2 = model2.predict(X_test)
        x2=model2.score(X_test,Y_test)
        self.Accuracy2.setText("Acurracy of Logistic Regression : {:.2f}".format(x2))
        #Confusion_matrix1
        cm=confusion_matrix(Y_test,Y_predmod2)
        plt.figure(figsize=(5,5))
        plt.title("Logistic Regression")
        fg=sn.heatmap(cm,annot=True)
        figure = fg.get_figure()
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        figure.savefig('Logistic Regression.jpg',dpi=400)

        #Desicion Tree
        from sklearn import tree
        model3 = tree.DecisionTreeClassifier()
        model3.fit(X_train,Y_train)
        Y_predmod3=model3.predict(X_test)
        x3=model3.score(X_test,Y_test)
        self.Accuracy3.setText("Acurracy of Desicion Tree : {:.2f}".format(x3))
        #Confusion_matrix1
        cm=confusion_matrix(Y_test,Y_predmod3)
        plt.figure(figsize=(5,5))
        plt.title("Desicion Tree")
        fg=sn.heatmap(cm,annot=True,cmap='BrBG')
        figure = fg.get_figure()
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        figure.savefig('Desicion Tree.jpg',dpi=400)

        #Support Vector Machines
        from sklearn.svm import SVC
        model4 = SVC(C=8)
        model4.fit(X_train,Y_train)
        Y_predmod4=model4.predict(X_test)
        x4=model4.score(X_test,Y_test)
        self.Accuracy4.setText("Acurracy of Support Vector Machines : {:.2f}".format(x4))
        #Confusion_matrix1
        cm=confusion_matrix(Y_test,Y_predmod4)
        plt.figure(figsize=(5,5))
        plt.title("Support Vector Machines")
        fg=sn.heatmap(cm,annot=True,cmap='Greens')
        figure = fg.get_figure()
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        figure.savefig('svm.jpg',dpi=400)

        #Random Forest
        from sklearn.ensemble import RandomForestClassifier
        model5 = RandomForestClassifier()
        model5.fit(X_train,Y_train)
        Y_predmod5=model5.predict(X_test)
        x5=model5.score(X_test,Y_test)
        self.Accuracy5.setText("Acurracy of Random Forest : {:.2f}".format(x5))
        #Confusion_matrix1
        cm=confusion_matrix(Y_test,Y_predmod5)
        plt.figure(figsize=(5,5))
        plt.title("Random Forest")
        fg=sn.heatmap(cm,annot=True,cmap='Oranges')
        figure = fg.get_figure()
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        figure.savefig('Random Forest.jpg',dpi=400)
        self.pushButton.clicked.connect(self.photo)
        
        #Xgbooster
        from xgboost import XGBClassifier
        model6=XGBClassifier()
        model6.fit(X_train,Y_train)
        Y_predmod6=model6.predict(X_test)
        x6=model6.score(X_test,Y_test)
        self.Accuracy6.setText("Acurracy of XGBClassifier : {:.2f}".format(x6))
        #Confusion_matrix1
        cm=confusion_matrix(Y_test,Y_predmod6)
        plt.figure(figsize=(5,5))
        plt.title("XGBClassifier")
        fg=sn.heatmap(cm,annot=True)
        figure = fg.get_figure()
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        figure.savefig('XGBClassifier.jpg',dpi=400)
        self.pushButton.clicked.connect(self.photo)

        #Neuralnetwork
        model7=keras.Sequential()
        model7.add(Dense(1,input_dim=22,activation='sigmoid'))
        model7.add(Dense(20,activation='sigmoid'))
        model7.add(Dense(12,activation='sigmoid'))
        model7.add(Dense(9,activation='sigmoid'))
        model7.add(Dense(5,activation='sigmoid'))
        model7.add(Dense(1,activation='sigmoid'))
        model7.compile(optimizer='adam',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
        model7.fit(X_train,Y_train,epochs=1000)
        _,x7=model7.evaluate(X_test,Y_test)
        prednn = model7.predict(X_test)
        for i,j in enumerate(prednn):
                if(j<0.5):
                        prednn[i]=0
                else:
                        prednn[i]=1
        prednn.reshape(1,39)
        predd = prednn.flatten()
        print(predd)
        self.Accuracy7.setText("Acurracy of Neural Network : {:.2f}".format(x7))
        #confusion Matrix
        cm=confusion_matrix(Y_test,predd)
        plt.figure(figsize=(5,5))
        plt.title("Neural Network")
        fg=sn.heatmap(cm,annot=True)
        figure = fg.get_figure()
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        figure.savefig('NeuralNetwork.jpg',dpi=400)
        self.pushButton.clicked.connect(self.photo)


    def photo(self):
            if self.comboBox1.currentIndex()==0:
                self.label.setPixmap(QtGui.QPixmap("Linear_Regression.jpg"))
            elif self.comboBox1.currentIndex()==1:
                self.label.setPixmap(QtGui.QPixmap("Logistic Regression.jpg"))
            elif self.comboBox1.currentIndex()==2:
                self.label.setPixmap(QtGui.QPixmap("Desicion Tree.jpg"))
            elif self.comboBox1.currentIndex()==3:
                self.label.setPixmap(QtGui.QPixmap("svm.jpg"))
            elif self.comboBox1.currentIndex()==4:
                self.label.setPixmap(QtGui.QPixmap("Random Forest.jpg"))
            elif self.comboBox1.currentIndex()==5:
                self.label.setPixmap(QtGui.QPixmap("XGBClassifier.jpg"))
            elif self.comboBox1.currentIndex()==6:
                self.label.setPixmap(QtGui.QPixmap("NeuralNetwork.jpg"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Parkinsons_Diseases = QtWidgets.QMainWindow()
    ui = Ui_Parkinsons_Diseases()
    ui.setupUi(Parkinsons_Diseases)
    Parkinsons_Diseases.show()
    sys.exit(app.exec_())

