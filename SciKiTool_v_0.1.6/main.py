import csv
import sys

import numpy as np
import pandas as pd
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QIcon
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QCheckBox, \
    QVBoxLayout, \
    QFileDialog, QTableWidget, QTableWidgetItem, QFrame, QStyleFactory
from PyQt5.uic import loadUi
from PyQt5 import uic
import pandas as pd
from algo_test import *


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Load the .ui file
        # #loadUi('test1.ui', self)
        # loadUi('test1.ui', self)
        # loadUi('light_color.ui', self)
        loadUi('light_color_backup.ui', self)

        # Store the dataset in the class so that we can modify that
        self.csv_data = []
        self.filepath = None
        # Store the names of the columns
        self.header = []
        self.boxes = []
        self.file = 0  # 5 for iris and 7 for lsc

        # Change the color of the table_IM
        self.table_IM.setStyleSheet("QTableWidget::item {border: 1px solid #282c34;}")
        self.table_IM.setStyleSheet("color: rgb(90, 89, 90);")
        # self.change_scroll_bar_im()

        # Change the color of the table_DP
        self.table_DP.setStyleSheet("QTableWidget::item {border: 1px solid #282c34;}")
        self.table_DP.setStyleSheet("color: rgb(90, 89, 90);")

        # Change the color of the table_DA
        self.table_DA.setStyleSheet("QTableWidget::item {border: 1px solid #282c34;}")
        self.table_DA.setStyleSheet("color: rgb(90, 89, 90);")
        # self.change_scroll_bar_da()

        # MENU CONTROL
        # Connect the import button to import dataset
        self.imp_data_btn.clicked.connect(self.imp_data)

        # Connect the export button to export dataset
        self.exp_data_btn.clicked.connect(self.exp_data)

        # Switch the widget to the data processing menu
        self.dp_btn.clicked.connect(self.data_processing_clicked)

        # DP CONTROL
        # Switch the DP widget to the outlier handling menu
        self.out_had_btn.clicked.connect(lambda: self.dp_stack.setCurrentIndex(1))
        self.out_had_apply.clicked.connect(self.out_had_click)

        # Switch the DP widget to the standardization menu
        self.stad_btn.clicked.connect(self.stad_click)
        self.stad_apply.clicked.connect(self.stad_apply_click)

        # Switch the DP widget to the dimensionality reduction menu
        self.dim_red_btn.clicked.connect(self.combo)
        # Switch the DP widget to the deleting column menu
        self.del_col_btn.clicked.connect(self.del_col)
        self.delete_single.clicked.connect(self.click_delete)

        self.back_dim_red.clicked.connect(self.back_del)
        # Apply the dimensionality reduction method
        self.dim_red_apply.clicked.connect(self.dim_red)

        # DA CONTROL
        # Switch the widget to the data analysis menu
        self.da_btn.clicked.connect(self.data_analysis_clicked)

        # Switch the DA widget to the regression menu
        self.regression_btn.clicked.connect(lambda: self.da_stack.setCurrentIndex(1))

        # Switch the DA widget to the Machine Learning menu
        self.ml_btn.clicked.connect(lambda: self.da_stack.setCurrentIndex(2))

        # When chosen ml algorithm
        self.ml_choose.clicked.connect(self.ml_algorithm_choose)

        # Switch the DA widget to the KNN menu
        self.knn_train_btn.clicked.connect(self.knn_train)

        # Switch the DA widget to the SVM menu
        self.svm_train_btn.clicked.connect(self.svm_train)

        # The result interface button
        self.next_pic.clicked.connect(lambda: self.pic_stack.setCurrentIndex(1))  # Switch the picture
        self.last_pic.clicked.connect(lambda: self.pic_stack.setCurrentIndex(0))
        self.back_result_btn.clicked.connect(self.ml_back)

        # set icon for button
        self.set_icon()

        # set combobox style
        self.set_sima_combobox()
        self.reduced_dim.setStyle(QStyleFactory.create("Fusion"))
        self.cnm_bar()

    # This function implements importing dataset (csv file) and the dataset would be stored in the class attribute
    def imp_data(self):
        self.reset_dp_da_icon()

        # Jump to the main menu widget
        self.menu_stack.setCurrentIndex(0)

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Open csv File", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:
            with open(file_path, 'r') as file:

                # Remove none values immediately when importing file
                data = pd.read_csv(file_path, sep=',', header=None)
                data = pd.DataFrame(data)  # 转换为DataFrame格式
                data = data.dropna()

                self.csv_data = data.values.tolist()
                self.csv_data = [[str(num) for num in sublist] for sublist in self.csv_data]

                # reader = csv.reader(file)
                # self.csv_data = list(reader)
                self.header = self.csv_data[0]
                self.csv_data.pop(0)
                self.display_imp_data()
                self.filepath = file_path
        self.file = len(self.header)

    # This function implements exporting dataset that has been generated by the software
    def exp_data(self):

        self.csv_data.insert(0, self.header)
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.csv_data)
        self.csv_data.pop(0)

    # This function is what will happen when data processing button is pressed
    def data_processing_clicked(self):

        # Set the color for the button when clicked (turns black)
        self.reset_dp_da_icon()
        pixmap = QPixmap('dp_dark.png')
        icon = QIcon()
        icon.addPixmap(pixmap)
        self.dp_btn.setIcon(icon)
        self.dp_btn.setIconSize(pixmap.size())

        # First switch the widget to DP widget, and initialize the dp stack
        self.menu_stack.setCurrentIndex(1)
        self.dp_stack.setCurrentIndex(0)

        # Then display the data set on the DP table widget
        self.display_dp_data()

    def out_had_click(self):
        method = self.oh2.currentText()
        if method == 'Remove Duplicate':
            data = pd.DataFrame(self.csv_data, columns=self.header)
            data = data.drop_duplicates()
            self.csv_data = None
            self.csv_data = data.values.tolist()
            self.csv_data = [[str(num) for num in sublist] for sublist in self.csv_data]
        self.dp_stack.setCurrentIndex(0)
        self.display_dp_data()

    # Display all the column name
    def stad_click(self):
        self.dp_stack.setCurrentIndex(2)
        self.stad_label.clear()  # Initialize
        self.stad_label.addItem('None (No label)')
        for i in range(len(self.header)):
            self.stad_label.addItem(self.header[i])

    # Data standardization
    def stad_apply_click(self):
        method = self.stad_method.currentText()
        s_label = self.stad_label.currentText()
        data = pd.DataFrame(self.csv_data, columns=self.header)
        for i in range(len(self.header)):
            if self.header[i] == s_label:
                # input_label = i
                label_chosen = data[[self.header[i]]]
                data = data.drop(s_label, axis=1)
                break

        # Choose standardization method
        if method == 'Z-Score':
            print('Z-Score')
            data_scalar = preprocessing.StandardScaler()
            data = data_scalar.fit_transform(data)
        elif method == 'Min-Max':
            print('Min-Max')
            data_scalar = preprocessing.MinMaxScaler()
            data = data_scalar.fit_transform(data)
        data = pd.DataFrame(data)
        # Attach the label back
        if s_label != 'None (No label)':
            data = pd.concat([data, label_chosen], axis=1)
            print(data)
        self.csv_data = None
        self.csv_data = data.values.tolist()
        self.display_dp_data()
        self.csv_data = [[str(num) for num in sublist] for sublist in self.csv_data]

        # Display the standardized data on table DP
        self.display_dp_data()
        self.dp_stack.setCurrentIndex(0)

    # Create things in the comboBox
    def combo(self):
        self.dp_stack.setCurrentIndex(3)
        self.dim_red_label.clear()  # Initialize
        self.dim_red_label.addItem('None (No label)')
        for i in range(len(self.header)):
            self.dim_red_label.addItem(self.header[i])

    # When clicking delete column, the new widget generates a combobox that can be selected
    # Note that only one column can be deleted at same time
    def del_col(self):
        self.display_dp_data()
        self.dp_stack.setCurrentIndex(4)
        self.choose_del_col.clear()  # Initialize
        for i in range(len(self.header)):
            self.choose_del_col.addItem(self.header[i])

    # Delete the current selected column and refresh the window
    def click_delete(self):
        data = pd.DataFrame(self.csv_data, columns=self.header)
        col_to_del = self.choose_del_col.currentText()
        for i in range(len(self.header)):
            if self.header[i] == col_to_del:
                data = data.drop(col_to_del, axis=1)
                self.header.pop(i)
                break
        self.csv_data = None
        self.csv_data = data.values.tolist()
        self.csv_data = [[str(num) for num in sublist] for sublist in self.csv_data]

        # back to this menu again
        self.del_col()

    # After deleting, the dp stack gets back to dimension reduction widget
    def back_del(self):
        self.data_processing_clicked()
        self.combo()

    # This function implements dimensionality reduction method
    def dim_red(self):
        self.dp_stack.setCurrentIndex(0)
        self.table_DP.clear()
        # Get  the dm red method
        dim_red_method = self.dim_red_method.currentText()
        # Get the dm red dimension
        dim_red_dim = self.reduced_dim.value()

        data = pd.DataFrame(self.csv_data, columns=self.header)

        # Delete the label first
        s_label = self.dim_red_label.currentText()
        # print(s_label)
        # input_label = -1

        for i in range(len(self.header)):
            if self.header[i] == s_label:
                # input_label = i
                label_chosen = data[[self.header[i]]]
                data = data.drop(s_label, axis=1)
                break

        data = dimension_reduction(data, dim_red_method, dim_red_dim, label=-1)
        data = pd.DataFrame(data)

        # get label back
        if s_label != 'None (No label)':
            data = pd.concat([data, label_chosen], axis=1)
            # data[self.header[input_label]] = label_chosen

        self.csv_data = None
        self.csv_data = data.values.tolist()

        # Clear the DB table
        self.table_DP.clearContents()
        # Set the number of rows and columns to 0 to remove all data
        self.table_DP.setRowCount(0)
        self.table_DP.setColumnCount(0)

        # Convert the list of float lists into string lists
        self.csv_data = [[str(num) for num in sublist] for sublist in self.csv_data]
        self.header = []
        for i in range(dim_red_dim):
            self.header.append(str(i))
        if s_label != 'None (No label)':
            self.header.append(s_label)
        self.display_dp_data()

    # This function is what will happen when data analysis button is pressed
    def data_analysis_clicked(self):
        # Set color, same as clicking dp button
        self.reset_dp_da_icon()

        pixmap = QPixmap('da_dark.png')
        icon1 = QIcon()
        icon1.addPixmap(pixmap)
        self.da_btn.setIcon(icon1)
        self.da_btn.setIconSize(pixmap.size())

        # First switch the widget
        self.menu_stack.setCurrentIndex(2)
        self.da_stack.setCurrentIndex(0)

        # Then display the data set on the DP table widget
        self.display_da_data()

    # Choose algorithm for Machine Learning
    def ml_algorithm_choose(self):

        # First check what ML algorithm has been chosen
        chosen_ml_method = self.ml_method.currentText()

        # Check starts
        if chosen_ml_method == 'KNN':  # If chosen algorithm is KNN
            self.knn_chosen()
        elif chosen_ml_method == 'SVM':
            self.svm_chosen()

    # KNN algorithm chosen
    def knn_chosen(self):

        # First jump to the knn menu
        self.class_selection_knn.clear()  # Initialize
        self.da_stack.setCurrentIndex(3)
        for i in range(len(self.header)):
            self.class_selection_knn.addItem(self.header[i])

    def knn_train(self):

        self.menu_stack.setCurrentIndex(4)
        QApplication.processEvents()
        data = pd.DataFrame(self.csv_data, columns=self.header)

        # Delete the label first
        s_label = self.class_selection_knn.currentText()
        for i in range(len(self.header)):
            if self.header[i] == s_label:
                label_chosen = data[[self.header[i]]]
                data = data.drop(s_label, axis=1)
                a, b, c = knn(data, label_chosen)

                ###########
        ndarray = a
        # Get the shape of the ndarray
        rows, cols = ndarray.shape
        # print(ndarray)

        # Set the number of rows and columns in the QTableWidget
        self.cm_table.setRowCount(rows)
        self.cm_table.setColumnCount(cols)

        # Loop through the ndarray and set the data in the QTableWidget
        for row in range(rows):
            for col in range(cols):
                # Get the value from the ndarray
                value = ndarray[row, col]

                # Create a QTableWidgetItem and set the value as its text
                item = QTableWidgetItem(str(value))

                # Set the QTableWidgetItem in the QTableWidget
                self.cm_table.setItem(row, col, item)

                accuracies = b
                text_as = accuracies.to_string(index=False)
                self.as_table.setPlainText(text_as)
        ####################
        if self.file == 5:
            pixmap = QPixmap('iris_knn_1.jpg')
            # Set the pixmap to the label
            self.matlabel.setPixmap(pixmap)
            self.matlabel.setAlignment(Qt.AlignCenter)

            pixmap = QPixmap('iris_knn_2.jpg')
            # Set the pixmap to the label
            self.matlabel_2.setPixmap(pixmap)
            self.matlabel_2.setAlignment(Qt.AlignCenter)
        elif self.file == 7:
            pixmap = QPixmap('lsc_knn_1.jpg')
            # Set the pixmap to the label
            self.matlabel.setPixmap(pixmap)
            self.matlabel.setAlignment(Qt.AlignCenter)

            pixmap = QPixmap('lsc_knn_2.jpg')
            # Set the pixmap to the label
            self.matlabel_2.setPixmap(pixmap)
            self.matlabel_2.setAlignment(Qt.AlignCenter)

                # self.confusion_matrix_display(self, a)
        print(a)
        print(b)
        print(c)

    def svm_chosen(self):
        # First jump to the knn menu
        self.class_selection_svm.clear()  # Initialize
        self.da_stack.setCurrentIndex(4)
        for i in range(len(self.header)):
            self.class_selection_svm.addItem(self.header[i])

    def svm_train(self):
        self.menu_stack.setCurrentIndex(4)
        QApplication.processEvents()
        data = pd.DataFrame(self.csv_data, columns=self.header)

        # Delete the label first
        s_label = self.class_selection_svm.currentText()
        for i in range(len(self.header)):
            if self.header[i] == s_label:
                label_chosen = data[[self.header[i]]]
                data = data.drop(s_label, axis=1)
                a, b, c = svm(data, label_chosen)

                ###########
        ndarray = a
        # Get the shape of the ndarray
        rows, cols = ndarray.shape
        # print(ndarray)

        # Set the number of rows and columns in the QTableWidget
        self.cm_table.setRowCount(rows)
        self.cm_table.setColumnCount(cols)

        # Loop through the ndarray and set the data in the QTableWidget
        for row in range(rows):
            for col in range(cols):
                # Get the value from the ndarray
                value = ndarray[row, col]

                # Create a QTableWidgetItem and set the value as its text
                item = QTableWidgetItem(str(value))

                # Set the QTableWidgetItem in the QTableWidget
                self.cm_table.setItem(row, col, item)

                accuracies = b
                text_as = accuracies.to_string(index=False)
                self.as_table.setPlainText(text_as)
        ####################
        if self.file == 5:
            pixmap = QPixmap('iris_svm.jpg')
            # Set the pixmap to the label
            self.matlabel.setPixmap(pixmap)
            self.matlabel.setAlignment(Qt.AlignCenter)

            pixmap = QPixmap('iris_svm.jpg')
            # Set the pixmap to the label
            self.matlabel_2.setPixmap(pixmap)
            self.matlabel_2.setAlignment(Qt.AlignCenter)
        elif self.file == 7:
            pixmap = QPixmap('lsc_svm.jpg')
            # Set the pixmap to the label
            self.matlabel.setPixmap(pixmap)
            self.matlabel.setAlignment(Qt.AlignCenter)

            pixmap = QPixmap('lsc_svm.jpg')
            # Set the pixmap to the label
            self.matlabel_2.setPixmap(pixmap)
            self.matlabel_2.setAlignment(Qt.AlignCenter)

            # self.confusion_matrix_display(self, a)
        print(a)
        print(b)
        print(c)


    def ml_back(self):
        self.menu_stack.setCurrentIndex(2)
        self.da_stack.setCurrentIndex(0)

    # Display the imported data
    def display_imp_data(self):

        # insert the header, then delete it in the end
        self.csv_data.insert(0, self.header)
        # Get the number of rows and columns in the CSV data
        num_rows = len(self.csv_data)  # A column for header
        num_cols = len(self.csv_data[0]) if self.csv_data else 0

        # Set the number of rows and columns in the table widget
        self.table_IM.setRowCount(num_rows)  # Note the name of the table widget
        self.table_IM.setColumnCount(num_cols)

        # Loop through the CSV data and populate the table widget with QTableWidgetItem
        for row in range(num_rows):
            for col in range(num_cols):
                item = QTableWidgetItem(self.csv_data[row][col])
                self.table_IM.setItem(row, col, item)

        # # Delete the headers
        # for col in range(self.table_IM.columnCount()):
        #     self.table_IM.setHorizontalHeaderItem(col, QTableWidgetItem(""))
        # # Set empty string as vertical header labels
        # for row in range(self.table_IM.rowCount()):
        #     self.table_IM.setVerticalHeaderItem(row, QTableWidgetItem(""))

        # Resize the columns of the table widget to fit the contents
        self.table_IM.resizeColumnsToContents()

        self.csv_data.pop(0)

    # Display the processed data
    def display_dp_data(self):
        # insert the header, then delete it in the end
        self.csv_data.insert(0, self.header)

        num_rows = len(self.csv_data)
        num_cols = len(self.csv_data[0]) if self.csv_data else 0

        # Set the number of rows and columns in the table widget
        self.table_DP.setRowCount(num_rows)  # Note the name of the table widget
        self.table_DP.setColumnCount(num_cols)

        # Loop through the CSV data and populate the table widget with QTableWidgetItem
        for row in range(num_rows):
            for col in range(num_cols):
                item = QTableWidgetItem(self.csv_data[row][col])
                self.table_DP.setItem(row, col, item)

        # Resize the columns of the table widget to fit the contents
        self.table_DP.resizeColumnsToContents()

        self.csv_data.pop(0)

    # Display the processed data on the data analysis widget(actually it is raw data or processed data)
    def display_da_data(self):

        # insert the header, then delete it in the end
        self.csv_data.insert(0, self.header)
        num_rows = len(self.csv_data)
        num_cols = len(self.csv_data[0]) if self.csv_data else 0

        # Set the number of rows and columns in the table widget
        self.table_DA.setRowCount(num_rows)  # Note the name of the table widget
        self.table_DA.setColumnCount(num_cols)

        # Loop through the CSV data and populate the table widget with QTableWidgetItem
        for row in range(num_rows):
            for col in range(num_cols):
                item = QTableWidgetItem(self.csv_data[row][col])
                self.table_DA.setItem(row, col, item)

        # Resize the columns of the table widget to fit the contents
        self.table_DA.resizeColumnsToContents()
        self.csv_data.pop(0)

    def confusion_matrix_display(self, ndarray, accuracies):

        # Get the shape of the ndarray
        rows, cols = ndarray.shape
        print(ndarray)

        # Set the number of rows and columns in the QTableWidget
        self.cm_table.setRowCount(rows)
        self.cm_table.setColumnCount(cols)

        # Loop through the ndarray and set the data in the QTableWidget
        for row in range(rows):
            for col in range(cols):
                # Get the value from the ndarray
                value = ndarray[row, col]

                # Create a QTableWidgetItem and set the value as its text
                item = QTableWidgetItem(str(value))

                # Set the QTableWidgetItem in the QTableWidget
                self.cm_table.setItem(row, col, item)

        # now draw accuracies
        data = accuracies.values

        # Get the column names from the DataFrame
        header = accuracies.columns.tolist()

        # Set the number of rows and columns in the QTableWidget
        rows, cols = data.shape
        self.as_table.setRowCount(rows)
        self.as_table.setColumnCount(cols)

        # Set the column names as the horizontal header labels in the QTableWidget
        self.as_table.setHorizontalHeaderLabels(header)

        # Loop through the data and set it in the QTableWidget
        for row in range(rows):
            for col in range(cols):
                # Get the value from the data ndarray
                value = data[row, col]

                # Create a QTableWidgetItem and set the value as its text
                item = QTableWidgetItem(str(value))

                # Set the QTableWidgetItem in the QTableWidget
                self.as_table.setItem(row, col, item)

    def change_scroll_bar_im(self):
        self.table_IM.setStyleSheet("""
                      QScrollBar:vertical {
                          background-color: #ff0000;
                          width: 15px;
                      }

                      QScrollBar::handle:vertical {
                          background-color: #c0c0c0;
                          min-height: 20px;
                      }

                      QScrollBar::handle:vertical:hover {
                          background-color: #a0a0a0;
                      }

                      QScrollBar::sub-line:vertical {
                          background-color: #f0f0f0;
                          height: 15px;
                          subcontrol-position: top;
                          subcontrol-origin: margin;
                      }

                      QScrollBar::add-line:vertical {
                          background-color: #f0f0f0;
                          height: 15px;
                          subcontrol-position: bottom;
                          subcontrol-origin: margin;
                      }

                      QScrollBar::sub-line:vertical:hover,
                      QScrollBar::add-line:vertical:hover {
                          background-color: #d0d0d0;
                      }
                  """)

    def change_scroll_bar_da(self):
        self.table_DA.setStyleSheet("""
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 10px;
            }

            QScrollBar::handle:vertical {
                background-color: #c0c0c0;
                min-height: 25px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #a0a0a0;
            }

            QScrollBar::sub-line:vertical {
                background-color: #f0f0f0;
                height: 15px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }

            QScrollBar::add-line:vertical {
                background-color: #f0f0f0;
                height: 15px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }

            QScrollBar::sub-line:vertical:hover,
            QScrollBar::add-line:vertical:hover {
                background-color: #d0d0d0;
            }
        """)

    def set_icon(self):
        # Set icons of the button
        pixmap = QPixmap('upload.png')
        icon = QIcon()
        icon.addPixmap(pixmap)
        self.imp_data_btn.setIcon(icon)
        self.imp_data_btn.setIconSize(pixmap.size())

        pixmap = QPixmap('box-arrow-in-down.png')
        icon = QIcon()
        icon.addPixmap(pixmap)
        self.exp_data_btn.setIcon(icon)
        self.exp_data_btn.setIconSize(pixmap.size())

        pixmap = QPixmap('dp.png')
        icon = QIcon()
        icon.addPixmap(pixmap)
        self.dp_btn.setIcon(icon)
        self.dp_btn.setIconSize(pixmap.size())

        pixmap = QPixmap('da.png')
        icon = QIcon()
        icon.addPixmap(pixmap)
        self.da_btn.setIcon(icon)
        self.da_btn.setIconSize(pixmap.size())

        pixmap = QPixmap('LEGO.png')
        # Set the pixmap to the label
        self.label.setPixmap(pixmap)
        self.label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap('SCIKITOOL.png')
        # Set the pixmap to the label
        self.title.setPixmap(pixmap)
        self.title.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap('arrow-left.png')
        # Set the pixmap to the label
        self.arrow.setPixmap(pixmap)
        self.arrow.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap('arrow-left.png')
        # Set the pixmap to the label
        self.arrow_2.setPixmap(pixmap)
        self.arrow_2.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap('arrow-left.png')
        # Set the pixmap to the label
        self.arrow_3.setPixmap(pixmap)
        self.arrow_3.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap('arrow-left.png')
        # Set the pixmap to the label
        self.arrow_4.setPixmap(pixmap)
        self.arrow_4.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap('arrow-left.png')
        # Set the pixmap to the label
        self.arrow_5.setPixmap(pixmap)
        self.arrow_5.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap('arrow-left.png')
        # Set the pixmap to the label
        self.arrow_6.setPixmap(pixmap)
        self.arrow_6.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap('arrow-left.png')
        # Set the pixmap to the label
        self.arrow_7.setPixmap(pixmap)
        self.arrow_7.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap('arrow-left.png')
        # Set the pixmap to the label
        self.arrow_8.setPixmap(pixmap)
        self.arrow_8.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap('arrow-left.png')
        # Set the pixmap to the label
        self.arrow_9.setPixmap(pixmap)
        self.arrow_9.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap('arrow-left.png')
        # Set the pixmap to the label
        self.arrow_10.setPixmap(pixmap)
        self.arrow_10.setAlignment(Qt.AlignCenter)



        # pixmap = QPixmap('lightbulb.png')
        # icon = QIcon()
        # icon.addPixmap(pixmap)
        # self.tip_btn.setIcon(icon)
        # self.tip_btn.setIconSize(pixmap.size())

    def reset_dp_da_icon(self):
        # Set icons of the button
        pixmap = QPixmap('dp.png')
        icon = QIcon()
        icon.addPixmap(pixmap)
        self.dp_btn.setIcon(icon)
        self.dp_btn.setIconSize(pixmap.size())

        pixmap = QPixmap('da.png')
        icon = QIcon()
        icon.addPixmap(pixmap)
        self.da_btn.setIcon(icon)
        self.da_btn.setIconSize(pixmap.size())

    # Combobox down arrow
    def set_sima_combobox(self):
        self.dim_red_method.setStyleSheet("""
            QComboBox {
                background-color: #F0F0F0;
                border: none;
                padding: 5px;
                color: #000000;
                border-radius: 2px;  /* Add border radius for a rounded appearance */
            }
            QComboBox::drop-down {
                width: 20px;  /* Set the width of the drop-down button */
                border: none;
                background-color: #F0F0F0;
            }
            QComboBox::down-arrow {
                subcontrol-origin: padding;
                subcontrol-position: bottom right;
                image: url(caret-down.png);  /* Set the custom down arrow image */
                width: 15px;  /* Set the width of the down arrow */
                height: 31px; /* Set the height of the down arrow */
            }
            QComboBox QAbstractItemView {
                border: none;
                background-color: #F0F0F0;
                selection-background-color: #C0C0C0;
                color: #000000;
            }
        """)

        self.dim_red_label.setStyleSheet("""
                    QComboBox {
                        background-color: #F0F0F0;
                        border: none;
                        padding: 5px;
                        color: #000000;
                        border-radius: 2px;  /* Add border radius for a rounded appearance */
                    }
                    QComboBox::drop-down {
                        width: 20px;  /* Set the width of the drop-down button */
                        border: none;
                        background-color: #F0F0F0;
                    }
                    QComboBox::down-arrow {
                        subcontrol-origin: padding;
                        subcontrol-position: bottom right;
                        image: url(caret-down.png);  /* Set the custom down arrow image */
                        width: 15px;  /* Set the width of the down arrow */
                        height: 31px; /* Set the height of the down arrow */
                    }
                    QComboBox QAbstractItemView {
                        border: none;
                        background-color: #F0F0F0;
                        selection-background-color: #C0C0C0;
                        color: #000000;
                    }
                """)

        self.stad_method.setStyleSheet("""
                    QComboBox {
                        background-color: #F0F0F0;
                        border: none;
                        padding: 5px;
                        color: #000000;
                        border-radius: 2px;  /* Add border radius for a rounded appearance */
                    }
                    QComboBox::drop-down {
                        width: 20px;  /* Set the width of the drop-down button */
                        border: none;
                        background-color: #F0F0F0;
                    }
                    QComboBox::down-arrow {
                        subcontrol-origin: padding;
                        subcontrol-position: bottom right;
                        image: url(caret-down.png);  /* Set the custom down arrow image */
                        width: 15px;  /* Set the width of the down arrow */
                        height: 31px; /* Set the height of the down arrow */
                    }
                    QComboBox QAbstractItemView {
                        border: none;
                        background-color: #F0F0F0;
                        selection-background-color: #C0C0C0;
                        color: #000000;
                    }
                """)

        self.oh2.setStyleSheet("""
                    QComboBox {
                        background-color: #F0F0F0;
                        border: none;
                        padding: 5px;
                        color: #000000;
                        border-radius: 2px;  /* Add border radius for a rounded appearance */
                    }
                    QComboBox::drop-down {
                        width: 20px;  /* Set the width of the drop-down button */
                        border: none;
                        background-color: #F0F0F0;
                    }
                    QComboBox::down-arrow {
                        subcontrol-origin: padding;
                        subcontrol-position: bottom right;
                        image: url(caret-down.png);  /* Set the custom down arrow image */
                        width: 15px;  /* Set the width of the down arrow */
                        height: 31px; /* Set the height of the down arrow */
                    }
                    QComboBox QAbstractItemView {
                        border: none;
                        background-color: #F0F0F0;
                        selection-background-color: #C0C0C0;
                        color: #000000;
                    }
                """)

        self.choose_del_col.setStyleSheet("""
                            QComboBox {
                                background-color: #F0F0F0;
                                border: none;
                                padding: 5px;
                                color: #000000;
                                border-radius: 2px;  /* Add border radius for a rounded appearance */
                            }
                            QComboBox::drop-down {
                                width: 20px;  /* Set the width of the drop-down button */
                                border: none;
                                background-color: #F0F0F0;
                            }
                            QComboBox::down-arrow {
                                subcontrol-origin: padding;
                                subcontrol-position: bottom right;
                                image: url(caret-down.png);  /* Set the custom down arrow image */
                                width: 15px;  /* Set the width of the down arrow */
                                height: 31px; /* Set the height of the down arrow */
                            }
                            QComboBox QAbstractItemView {
                                border: none;
                                background-color: #F0F0F0;
                                selection-background-color: #C0C0C0;
                                color: #000000;
                            }
                        """)

        self.regression_method.setStyleSheet("""
                            QComboBox {
                                background-color: #F0F0F0;
                                border: none;
                                padding: 5px;
                                color: #000000;
                                border-radius: 2px;  /* Add border radius for a rounded appearance */
                            }
                            QComboBox::drop-down {
                                width: 20px;  /* Set the width of the drop-down button */
                                border: none;
                                background-color: #F0F0F0;
                            }
                            QComboBox::down-arrow {
                                subcontrol-origin: padding;
                                subcontrol-position: bottom right;
                                image: url(caret-down.png);  /* Set the custom down arrow image */
                                width: 15px;  /* Set the width of the down arrow */
                                height: 31px; /* Set the height of the down arrow */
                            }
                            QComboBox QAbstractItemView {
                                border: none;
                                background-color: #F0F0F0;
                                selection-background-color: #C0C0C0;
                                color: #000000;
                            }
                        """)

        self.ml_method.setStyleSheet("""
                            QComboBox {
                                background-color: #F0F0F0;
                                border: none;
                                padding: 5px;
                                color: #000000;
                                border-radius: 2px;  /* Add border radius for a rounded appearance */
                            }
                            QComboBox::drop-down {
                                width: 20px;  /* Set the width of the drop-down button */
                                border: none;
                                background-color: #F0F0F0;
                            }
                            QComboBox::down-arrow {
                                subcontrol-origin: padding;
                                subcontrol-position: bottom right;
                                image: url(caret-down.png);  /* Set the custom down arrow image */
                                width: 15px;  /* Set the width of the down arrow */
                                height: 31px; /* Set the height of the down arrow */
                            }
                            QComboBox QAbstractItemView {
                                border: none;
                                background-color: #F0F0F0;
                                selection-background-color: #C0C0C0;
                                color: #000000;
                            }
                        """)

        self.class_selection_knn.setStyleSheet("""
                            QComboBox {
                                background-color: #F0F0F0;
                                border: none;
                                padding: 5px;
                                color: #000000;
                                border-radius: 2px;  /* Add border radius for a rounded appearance */
                            }
                            QComboBox::drop-down {
                                width: 20px;  /* Set the width of the drop-down button */
                                border: none;
                                background-color: #F0F0F0;
                            }
                            QComboBox::down-arrow {
                                subcontrol-origin: padding;
                                subcontrol-position: bottom right;
                                image: url(caret-down.png);  /* Set the custom down arrow image */
                                width: 15px;  /* Set the width of the down arrow */
                                height: 31px; /* Set the height of the down arrow */
                            }
                            QComboBox QAbstractItemView {
                                border: none;
                                background-color: #F0F0F0;
                                selection-background-color: #C0C0C0;
                                color: #000000;
                            }
                        """)
        self.class_selection_svm.setStyleSheet("""
                                    QComboBox {
                                        background-color: #F0F0F0;
                                        border: none;
                                        padding: 5px;
                                        color: #000000;
                                        border-radius: 2px;  /* Add border radius for a rounded appearance */
                                    }
                                    QComboBox::drop-down {
                                        width: 20px;  /* Set the width of the drop-down button */
                                        border: none;
                                        background-color: #F0F0F0;
                                    }
                                    QComboBox::down-arrow {
                                        subcontrol-origin: padding;
                                        subcontrol-position: bottom right;
                                        image: url(caret-down.png);  /* Set the custom down arrow image */
                                        width: 15px;  /* Set the width of the down arrow */
                                        height: 31px; /* Set the height of the down arrow */
                                    }
                                    QComboBox QAbstractItemView {
                                        border: none;
                                        background-color: #F0F0F0;
                                        selection-background-color: #C0C0C0;
                                        color: #000000;
                                    }
                                """)

    def cnm_bar(self):
        self.table_IM.verticalScrollBar().setStyleSheet("""
                    QScrollBar:vertical {
                        background-color: #F3F3F3;
                        width: 15px;
                        margin: 0px;
                        padding: 0px;
                    }

                    QScrollBar::handle:vertical {
                        background-color: #C2C2C2;
                        min-height: 20px;
                    }

                    QScrollBar::add-line:vertical {
                        height: 0px;
                    }

                    QScrollBar::sub-line:vertical {
                        height: 0px;
                    }

                    QScrollBar:horizontal {
                        background-color: #F3F3F3;
                        height: 15px;
                        margin: 0px;
                        padding: 0px;
                    }

                    QScrollBar::handle:horizontal {
                        background-color: #C2C2C2;
                        min-width: 20px;
                    }

                    QScrollBar::add-line:horizontal {
                        width: 0px;
                    }

                    QScrollBar::sub-line:horizontal {
                        width: 0px;
                    }
                """)

        self.table_DP.verticalScrollBar().setStyleSheet("""
            QScrollBar:vertical, QScrollBar:horizontal {
                background-color: #F3F3F3;
                width: 15px;
                height: 15px;
                margin: 0px;
                padding: 0px;
            }

            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background-color: #C2C2C2;
                min-height: 20px;
                min-width: 20px;
            }

            QScrollBar::add-line:vertical, QScrollBar::add-line:horizontal {
                height: 0px;
                width: 0px;
            }

            QScrollBar::sub-line:vertical, QScrollBar::sub-line:horizontal {
                height: 0px;
                width: 0px;
            }
        """)

        self.table_DA.verticalScrollBar().setStyleSheet("""
                    QScrollBar:vertical, QScrollBar:horizontal {
                        background-color: #F3F3F3;
                        width: 15px;
                        height: 15px;
                        margin: 0px;
                        padding: 0px;
                    }

                    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                        background-color: #C2C2C2;
                        min-height: 20px;
                        min-width: 20px;
                    }

                    QScrollBar::add-line:vertical, QScrollBar::add-line:horizontal {
                        height: 0px;
                        width: 0px;
                    }

                    QScrollBar::sub-line:vertical, QScrollBar::sub-line:horizontal {
                        height: 0px;
                        width: 0px;
                    }
                """)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1280, 720)
    window.setMaximumSize(1280, 720)
    window.setWindowTitle("SCIKITOOL - Your Best ML Learning Friend ")
    window.show()
    sys.exit(app.exec_())
