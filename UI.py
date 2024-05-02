from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QErrorMessage
import sys
import numpy as np
import joblib

import warnings
warnings.filterwarnings('ignore')

ui_file = "UI/UI.ui"

class UI_mainWindow(QMainWindow):
    
    def __init__(self):
        super(UI_mainWindow, self).__init__()
        uic.loadUi(ui_file, self)  # Загрузка UI-файла
        self.setWindowTitle("Systola")
        self.error_dialog = QErrorMessage()
        
        # Добавить функционал кнопке calculate:
        self.calculate_button.clicked.connect(self.calc_button_func)
        
        self.scaler = joblib.load("Normalization/scaler.pkl")
        self.model = joblib.load("Model/model.pkl")
        
        self.best_thrsh = 0.4124124124124124
        
    def calc_button_func(self):
        try:
            self.age = float(self.age_input.text())
            self.BMI = float(self.BMI_input.text())
            self.alcohol = float(self.alcohol_cb.currentIndex())
            self.sex = float(self.sex_cb.currentIndex())
            self.smoke = float(self.smoke_cb.currentIndex())
            self.sport = float(self.sport_cb.currentIndex())
            self.cholesterol = float(self.cholesterol_input.text())
            self.dis_b_p = float(self.dis_b_p_input.text())
            self.glucose = float(self.glucose_input.text())
            self.sys_b_p = float(self.sys_b_p_input.text())
        except ValueError:  # Обработка исключения конкретно ValueError
            self.error_dialog.showMessage('Неправильно введены данные')
            self.statusBar().showMessage('Ошибка ввода данных')
            return
        self.result_label.setText(f"Результат: {'У вас есть риск сердечно-сосудистого заболевания' if self.calculate() else 'У вас нет риска сердечно-сосудистого заболевания'}")
        
    def calculate(self):
        values = np.array([self.age, self.sex, self.BMI, self.sys_b_p, self.dis_b_p, self.cholesterol, self.glucose, self.smoke, self.alcohol, self.sport])
        
        values = self.scaler.transform(np.array([values]))
        predictions = self.model.predict(values)
        
        return predictions > self.best_thrsh
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = UI_mainWindow()
    mainwindow.show()

    sys.exit(app.exec_())
