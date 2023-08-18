import os
import logging
import warnings
import pet
import glob
import electrofacies as ef
from tqdm import tqdm
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QCheckBox, QVBoxLayout, QWidget, QFileDialog

class ProcessingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.source_folder_label = QLabel('Source Folder:')
        self.source_folder_input = QLineEdit()
        self.source_folder_button = QPushButton('Browse')
        self.source_folder_button.clicked.connect(self.browseSourceFolder)

        self.dest_folder_label = QLabel('Destination Folder:')
        self.dest_folder_input = QLineEdit()
        self.dest_folder_button = QPushButton('Browse')
        self.dest_folder_button.clicked.connect(self.browseDestFolder)

        self.precondition_checkbox = QCheckBox('Apply Precondition')
        self.fluidprop_checkbox = QCheckBox('Apply Fluid Properties')
        self.multimineral_checkbox = QCheckBox('Apply Multimineral Model')
        self.electrofacies_checkbox = QCheckBox('Apply Electrofacies')

        self.process_button = QPushButton('Process')
        self.process_button.clicked.connect(self.processFiles)

        layout = QVBoxLayout()
        layout.addWidget(self.source_folder_label)
        layout.addWidget(self.source_folder_input)
        layout.addWidget(self.source_folder_button)
        layout.addWidget(self.dest_folder_label)
        layout.addWidget(self.dest_folder_input)
        layout.addWidget(self.dest_folder_button)
        layout.addWidget(self.precondition_checkbox)
        layout.addWidget(self.fluidprop_checkbox)
        layout.addWidget(self.multimineral_checkbox)
        layout.addWidget(self.electrofacies_checkbox)
        layout.addWidget(self.process_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.setWindowTitle('Log Processing GUI')
        self.setGeometry(100, 100, 400, 300)

    def browseSourceFolder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Source Folder')
        self.source_folder_input.setText(folder)

    def browseDestFolder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
        self.dest_folder_input.setText(folder)

    def processFiles(self):
        source_folder = self.source_folder_input.text()
        dest_folder = self.dest_folder_input.text()

        precondition = self.precondition_checkbox.isChecked()
        fluidprop = self.fluidprop_checkbox.isChecked()
        multimineral = self.multimineral_checkbox.isChecked()
        electrofacies = self.electrofacies_checkbox.isChecked()

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        las_files = glob.glob(os.path.join(source_folder, '*.las'))

        # Initialize a status bar for processing
        progress_bar = tqdm(total=len(las_files), desc="Processing LAS files")

        processed_files = []

        for las_file in las_files:
            try:
                log = pet.Log(las_file)

                if precondition:
                    log.precondition()
                if fluidprop:
                    log.fluid_properties()
                if multimineral:
                    log.multimineral_model()

                well_name = log.well['UWI'].value.replace('.', '')
                processed_las_file = os.path.join(dest_folder, f'{well_name}_processed.las')
                log.write(processed_las_file)

                processed_files.append(processed_las_file)

            except Exception as e:
                print(f"An error occurred while processing {las_file}: {str(e)}")
                logging.error(f"An error occurred while processing {las_file}: {str(e)}")

            progress_bar.update()

        progress_bar.close()

        if electrofacies:
            logs = [pet.Log(x) for x in processed_files]

            progress_bar_ef = tqdm(total=len(logs), desc="Applying Electrofacies")
            combined_logs = ef.electrofacies(logs=logs)

            for i, log in enumerate(combined_logs):
                log.write(processed_files[i])  # Overwrite the processed file with electrofacies results
                progress_bar_ef.update()

            progress_bar_ef.close()

        print("Processing completed!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ProcessingGUI()
    gui.show()
    sys.exit(app.exec_())
