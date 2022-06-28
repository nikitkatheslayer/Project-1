from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import numpy
import matplotlib.pyplot as plt

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(560, 436)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 561, 411))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(110, 10, 341, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(40, 70, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton_7 = QtWidgets.QPushButton(self.tab)
        self.pushButton_7.setGeometry(QtCore.QRect(180, 190, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 150, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(40, 150, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.textEdit_2 = QtWidgets.QTextEdit(self.tab)
        self.textEdit_2.setGeometry(QtCore.QRect(350, 100, 151, 31))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit = QtWidgets.QTextEdit(self.tab)
        self.textEdit.setGeometry(QtCore.QRect(350, 60, 151, 31))
        self.textEdit.setObjectName("textEdit")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(40, 110, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.textEdit_7 = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit_7.setGeometry(QtCore.QRect(350, 210, 151, 31))
        self.textEdit_7.setObjectName("textEdit_7")
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        self.label_6.setGeometry(QtCore.QRect(50, 60, 161, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.textEdit_6 = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit_6.setGeometry(QtCore.QRect(350, 170, 151, 31))
        self.textEdit_6.setObjectName("textEdit_6")
        self.textEdit_3 = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit_3.setGeometry(QtCore.QRect(350, 50, 151, 31))
        self.textEdit_3.setObjectName("textEdit_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_4.setGeometry(QtCore.QRect(320, 260, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.textEdit_5 = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit_5.setGeometry(QtCore.QRect(350, 130, 151, 31))
        self.textEdit_5.setObjectName("textEdit_5")
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.label_4.setGeometry(QtCore.QRect(170, 10, 211, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_5.setGeometry(QtCore.QRect(190, 300, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_8 = QtWidgets.QLabel(self.tab_2)
        self.label_8.setGeometry(QtCore.QRect(50, 140, 151, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.textEdit_4 = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit_4.setGeometry(QtCore.QRect(350, 90, 151, 31))
        self.textEdit_4.setObjectName("textEdit_4")
        self.label_10 = QtWidgets.QLabel(self.tab_2)
        self.label_10.setGeometry(QtCore.QRect(50, 220, 251, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_9 = QtWidgets.QLabel(self.tab_2)
        self.label_9.setGeometry(QtCore.QRect(50, 180, 151, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        self.label_7.setGeometry(QtCore.QRect(50, 100, 171, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 260, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 560, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.pushButton.clicked.connect(self.otjig)
        self.pushButton_3.clicked.connect(self.muravei)
        
        self.pushButton_5.clicked.connect(self.cleanMyravei)
        self.pushButton_7.clicked.connect(self.cleanOtjig)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Метод имитации отжига"))
        self.label_2.setText(_translate("MainWindow", "Температра"))
        self.pushButton_7.setText(_translate("MainWindow", "Очистить"))
        self.pushButton_2.setText(_translate("MainWindow", "Показать график"))
        self.pushButton.setText(_translate("MainWindow", "Расчитать"))
        self.label_3.setText(_translate("MainWindow", "Изменение температуры"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Метод отжига"))
        self.label_6.setText(_translate("MainWindow", "Количество вершин"))
        self.pushButton_4.setText(_translate("MainWindow", "Показать график"))
        self.label_4.setText(_translate("MainWindow", "Метод муравья"))
        self.pushButton_5.setText(_translate("MainWindow", "Очистить"))
        self.label_8.setText(_translate("MainWindow", "Коэффицент alpha"))
        self.label_10.setText(_translate("MainWindow", "Скорость испарения феромона"))
        self.label_9.setText(_translate("MainWindow", "Коэффицент beta"))
        self.label_7.setText(_translate("MainWindow", "Количество муравьев"))
        self.pushButton_3.setText(_translate("MainWindow", "Расчитать"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Метод муравья"))
        
    def cleanOtjig(self):
        self.textEdit.clear()
        self.textEdit_2.clear()
        
    def cleanMyravei(self):
        self.textEdit_3.clear()
        self.textEdit_4.clear()
        self.textEdit_5.clear()
        self.textEdit_6.clear()
        self.textEdit_7.clear()
        
    def muravei(self):
        import numpy as np
        from scipy import spatial
        
        class ACO_TSP:  # класс алгоритма муравьиной колонии для решения задачи коммивояжёра
            def __init__(self, func, n_dim, size_pop=10, max_iter=20, distance_matrix=None, alpha = int(self.textEdit_5.toPlainText()), beta = int(self.textEdit_6.toPlainText()), rho = float(self.textEdit_7.toPlainText())):
                self.func = func
                self.n_dim = n_dim  # количество городов
                self.size_pop = size_pop  # количество муравьёв
                self.max_iter = max_iter  # количество итераций
                self.alpha = alpha  # коэффициент важности феромонов в выборе пути
                self.beta = beta  # коэффициент значимости расстояния
                self.rho = rho  # скорость испарения феромонов

                self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(n_dim, n_dim))

                # Матрица феромонов, обновляющаяся каждую итерацию
                self.Tau = np.ones((n_dim, n_dim))
                # Путь каждого муравья в определённом поколении
                self.Table = np.zeros((size_pop, n_dim)).astype(int)
                self.y = None  # Общее расстояние пути муравья в определённом поколении
                self.generation_best_X, self.generation_best_Y = [], [] # фиксирование лучших поколений
                self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y
                self.best_x, self.best_y = None, None
                
            def run(self, max_iter=None):
                self.max_iter = max_iter or self.max_iter
                for i in range(self.max_iter):
                    # вероятность перехода без нормализации
                    prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_distance) ** self.beta
                    for j in range(self.size_pop):  # для каждого муравья
                        # точка начала пути (она может быть случайной, это не имеет значения)
                        self.Table[j, 0] = 0
                        for k in range(self.n_dim - 1):  # каждая вершина, которую проходят муравьи
                            # точка, которая была пройдена и не может быть пройдена повторно
                            taboo_set = set(self.Table[j, :k + 1])
                            # список разрешённых вершин, из которых будет происходить выбор
                            allow_list = list(set(range(self.n_dim)) - taboo_set)
                            prob = prob_matrix[self.Table[j, k], allow_list]
                            prob = prob / prob.sum() # нормализация вероятности
                            next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                            self.Table[j, k + 1] = next_point

                    # рассчёт расстояния
                    y = np.array([self.func(i) for i in self.Table])

                    # фиксация лучшего решения
                    index_best = y.argmin()
                    x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
                    self.generation_best_X.append(x_best)
                    self.generation_best_Y.append(y_best)

                    # подсчёт феромона, который будет добавлен к ребру
                    delta_tau = np.zeros((self.n_dim, self.n_dim))
                    for j in range(self.size_pop):  # для каждого муравья
                        for k in range(self.n_dim - 1):  # для каждой вершины
                            # муравьи перебираются из вершины n1 в вершину n2
                            n1, n2 = self.Table[j, k], self.Table[j, k + 1]
                            delta_tau[n1, n2] += 1 / y[j]  # нанесение феромона
                        # муравьи ползут от последней вершины обратно к первой
                        n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]
                        delta_tau[n1, n2] += 1 / y[j]  # нанесение феромона

                    self.Tau = (1 - self.rho) * self.Tau + delta_tau

                best_generation = np.array(self.generation_best_Y).argmin()
                self.best_x = self.generation_best_X[best_generation]
                self.best_y = self.generation_best_Y[best_generation]
                return self.best_x, self.best_y
            #обучение модели
            fit = run
            
        num_points = int(self.textEdit_3.toPlainText()) #количество вершин
        
        if (int(self.textEdit_3.toPlainText()) < 0) | (int(self.textEdit_4.toPlainText()) < 0):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка!")
            msg.setInformativeText('Введены отрицательные числа')
            msg.setWindowTitle("info")
            msg.exec_()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Отлично!")
            msg.setInformativeText('Данные корректны!')
            msg.setWindowTitle("info")
            msg.exec_()
            
            print("Количество городов")
            print(int(self.textEdit_3.toPlainText()))
            print("Количество муравьев")
            print(int(self.textEdit_4.toPlainText()))
            
            points_coordinate = np.random.rand(num_points, 2)
            # вычисление матрицы расстояний между вершин
            distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
        
            # вычисление длины пути
            def cal_total_distance(routine):
                num_points, = routine.shape
                return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

            def main():
                # создание объекта алгоритма муравьиной колонии
                aca = ACO_TSP(func=cal_total_distance, n_dim=num_points,
                              size_pop= int(self.textEdit_4.toPlainText()),  # количество муравьёв
                              max_iter=1, distance_matrix=distance_matrix)
                best_x, best_y = aca.run()

                # Вывод результатов на экран
                fig, ax = plt.subplots(1, 2)
                best_points_ = np.concatenate([best_x, [best_x[0]]])
                best_points_coordinate = points_coordinate[best_points_, :]
                for index in range(0, len(best_points_)):
                    ax[1].annotate(best_points_[index], (best_points_coordinate[index, 0], best_points_coordinate[index, 1]))
                ax[0].plot(points_coordinate[:, 0],
                           points_coordinate[:, 1], 'o-r')
                plt.title(r'Метод муравья', fontsize=20)
                ax[1].plot(best_points_coordinate[:, 0],
                           best_points_coordinate[:, 1], 'o-r')
                # изменение размера графиков
                plt.rcParams['figure.figsize'] = [10,5]
                #сохранение плота
                plt.savefig("plot1.png", format='png', dpi=150, bbox_inches='tight')
                
            
            if __name__ == "__main__":
                main() # выполнение кода
    #метод отжига
    def otjig(self):
        
        class Coordinate:
            #вводим координаты
            def __init__(self, x, y):
                self.x = x
                self.y = y
                
            #метод для нахождения дистанции между соседями
            def get_distance(a, b):
                return numpy.sqrt(numpy.abs(a.x - b.x) + numpy.abs(a.y - b.y))
            
            #метод для нахождения общей дистанции между всеми координатами
            def get_total_distance(coords):
                dist = 0
                for first, second in zip(coords[:-1], coords[1:]):
                    dist += Coordinate.get_distance(first, second)
                dist += Coordinate.get_distance(coords[0], coords[-1])
                return dist
        
        #параметры температуры и ее изменения
        T = float(self.textEdit.toPlainText())
        factor = float(self.textEdit_2.toPlainText())
        
        if (factor < 0) | (T < 0):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка!")
            msg.setInformativeText('Введены некорректные числа')
            msg.setWindowTitle("info")
            msg.exec_()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Отлично!")
            msg.setInformativeText('Данные корректны!')
            msg.setWindowTitle("info")
            msg.exec_()
            
            print("Температура")
            print(T)
            print("Изменение температуры")
            print(factor)
            
            if __name__ == "__main__":
                #Заполнение координат
                coords = []
                for i in range(25): #Количество вершин\можно задать
                    coords.append(Coordinate(numpy.random.uniform(), numpy.random.uniform()))
                
                #Построение плота
                fig = plt.figure(figsize = (10, 5))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                for first, second in zip(coords[:-1], coords[1:]):
                    ax1.plot([first.x, second.x], [first.y, second.y], "b")
                ax1.plot([coords[0].x, coords[-1].x], [coords[0].y, coords[-1].y], "b")
                for c in coords:
                    ax1.plot(c.x, c.y, "ro")
                
                #Алгоритм имитации отжига
                cost0 = Coordinate.get_total_distance(coords)
                
                for i in range(500):
                    #print(i, "cost =", cost0)
                    
                    T = T * factor
                    for j in range(100): #Set
                        #Обмен двух вершин и получение нового соседнего
                        r1, r2 = numpy.random.randint(0, len(coords), size=2)
                        
                        temp = coords[r1]
                        coords[r1] = coords[r2]
                        coords[r2] = temp
                        
                        #Получение новой координаты
                        cost1 = Coordinate.get_total_distance(coords)
                        
                        #Нахождение оптимального пути
                        if cost1 < cost0:
                            cost0 = cost1
                        else:
                            x = numpy.random.uniform()
                            if x < numpy.exp((cost0-cost1)/T):
                                cost0 = cost1
                            else:
                                temp = coords[r1]
                                coords[r1] = coords[r2]
                                coords[r2] = temp
                
                #Вывод конечного плота
                for first, second in zip(coords[:-1], coords[1:]):
                    ax2.plot([first.x, second.x], [first.y, second.y], "b")
                ax2.plot([coords[0].x, coords[-1].x], [coords[0].y, coords[-1].y], "b")
                for c in coords:
                    ax2.plot(c.x, c.y, "ro")
                print(first, second)
                plt.title(r'Метод имитации отжига', fontsize=20)
                #Сохранение плота
                plt.savefig("plot.png", format='png', dpi=150, bbox_inches='tight')
            
#Параметры для окна и загрузка в него плота(Отжиг)
class ClssDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(ClssDialog, self).__init__(parent)
        from PyQt5.QtWidgets import (QHBoxLayout, QLabel)
        from PyQt5.QtGui import QPixmap
        
        hbox = QHBoxLayout(self)
        pixmap = QPixmap("plot.png")
        lbl = QLabel(self)
        lbl.setPixmap(pixmap)
        hbox.addWidget(lbl)
        self.setLayout(hbox)
        
        self.move(100, 200)
        self.setWindowTitle('График')
        self.show()

#Параметры для окна и загрузка в него плота(Муравей)
class ClssDialog1(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(ClssDialog1, self).__init__(parent)
        from PyQt5.QtWidgets import (QHBoxLayout, QLabel)
        from PyQt5.QtGui import QPixmap
        
        hbox = QHBoxLayout(self)
        pixmap = QPixmap("plot1.png")
        lbl = QLabel(self)
        lbl.setPixmap(pixmap)
        hbox.addWidget(lbl)
        self.setLayout(hbox)
        
        self.move(100, 200)
        self.setWindowTitle('График')
        self.show()
        
#Класс для вывода окна с графами
class MyWin(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow() 
        self.ui.setupUi(self)
        
        #Нажатие кнопок, при котором будет открываться новое окно с графами
        self.ui.pushButton_2.clicked.connect(self.openDialog)
        self.ui.pushButton_4.clicked.connect(self.openDialog1)

    #Методы с помощью которого реализуется нажатие на кнопку
    def openDialog(self):
        pass
        dialog = ClssDialog(self)
        dialog.exec_()
        
    def openDialog1(self):
        pass
        dialog = ClssDialog1(self)
        dialog.exec_()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MyWin()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

