import os
import logging
import warnings
import pet
import glob
import electrofacies as ef
from tqdm import tqdm
import sys
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QCheckBox, QVBoxLayout, QWidget, QFileDialog, QProgressBar, QStatusBar

class ProcessingThread(threading.Thread):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def run(self):
        source_folder = self.parent.source_folder_input.text()
        dest_folder = self.parent.dest_folder_input.text()

        precondition = self.parent.precondition_checkbox.isChecked()
        fluidprop = self.parent.fluidprop_checkbox.isChecked()
        multimineral = self.parent.multimineral_checkbox.isChecked()
        electrofacies = self.parent.electrofacies_checkbox.isChecked()

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        las_files = glob.glob(os.path.join(source_folder, '*.las'))

        processed_files = []

        for index, las_file in enumerate(las_files):
            try:
                log = pet.Log(las_file)

                if precondition:
                    log.precondition(drho_matrix=self.parent.drho_matrix, n=self.parent.n)
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

            self.parent.update_progress(index + 1, len(las_files))
            self.parent.update_status_message(f"Processing {index + 1}/{len(las_files)} files")

        if electrofacies:
            logs = [pet.Log(x) for x in processed_files]
            self.parent.update_progress(len(las_files) + 1, len(las_files) + 1)

            curves = []
            if self.parent.nphi_checkbox.isChecked():
                curves.append('NPHI')
            if self.parent.rhob_checkbox.isChecked():
                curves.append('RHOB')
            if self.parent.ild_checkbox.isChecked():
                curves.append('ILD')
            if self.parent.gr_checkbox.isChecked():
                curves.append('GR')
            if self.parent.pe_checkbox.isChecked():
                curves.append('PE')
            if self.parent.dt_checkbox.isChecked():
                curves.append('DT')
            
            combined_logs = ef.electrofacies(logs=logs, curves=curves, n_clusters=self.parent.n_clusters)

            for i, log in enumerate(combined_logs):
                log.write(processed_files[i])

            self.parent.update_status_message(f"Processing Electrofacies (Step {len(las_files) + 1}/{len(las_files) + 1})")

        self.parent.processing_completed()
        self.parent.update_status_message("Processing completed!")

class ProcessingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.drho_matrix = 2.71
        self.n = 7
        self.n_clusters = 6
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

        self.drho_matrix_label = QLabel('drho_matrix: for sandstone = 2.65, for dolomite = 2.87, for limestone = 2.71')
        self.drho_matrix_input = QLineEdit(str(self.drho_matrix))

        self.n_label = QLabel('n: apply lfilter lower n less smoothing higher n more smoothing')
        self.n_input = QLineEdit(str(self.n))

        self.n_clusters_label = QLabel('n_clusters:')
        self.n_clusters_input = QLineEdit(str(self.n_clusters))

        self.precondition_checkbox = QCheckBox('Apply Precondition')
        self.fluidprop_checkbox = QCheckBox('Apply Fluid Properties')
        self.multimineral_checkbox = QCheckBox('Apply Multimineral Model')
        self.electrofacies_checkbox = QCheckBox('Apply Electrofacies')

        self.curves_label = QLabel('Curves: Check the curves you want to use for electrofacies')
        self.nphi_checkbox = QCheckBox('NPHI')
        self.nphi_checkbox.setChecked(True)  # Check this box by default
        self.rhob_checkbox = QCheckBox('RHOB')
        self.rhob_checkbox.setChecked(True)  # Check this box by default
        self.ild_checkbox = QCheckBox('ILD')
        self.ild_checkbox.setChecked(True)  # Check this box by default
        self.gr_checkbox = QCheckBox('GR')
        self.pe_checkbox = QCheckBox('PE')
        self.dt_checkbox = QCheckBox('DT')

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
        layout.addWidget(self.drho_matrix_label)
        layout.addWidget(self.drho_matrix_input)
        layout.addWidget(self.n_label)
        layout.addWidget(self.n_input)
        layout.addWidget(self.fluidprop_checkbox)
        layout.addWidget(self.multimineral_checkbox)
        layout.addWidget(self.electrofacies_checkbox)
        layout.addWidget(self.n_clusters_label)
        layout.addWidget(self.n_clusters_input)
        layout.addWidget(self.curves_label)
        layout.addWidget(self.nphi_checkbox)
        layout.addWidget(self.rhob_checkbox)
        layout.addWidget(self.ild_checkbox)
        layout.addWidget(self.gr_checkbox)
        layout.addWidget(self.pe_checkbox)
        layout.addWidget(self.dt_checkbox)
        layout.addWidget(self.process_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.setWindowTitle('Log Processing GUI')
        self.setGeometry(200, 200, 400, 500)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.status_bar.addWidget(self.progress_bar)

    def browseSourceFolder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Source Folder')
        self.source_folder_input.setText(folder)

    def browseDestFolder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
        self.dest_folder_input.setText(folder)

    def processFiles(self):
        self.process_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(1)

        self.drho_matrix = float(self.drho_matrix_input.text())
        self.n = int(self.n_input.text())
        self.n_clusters = int(self.n_clusters_input.text())

        self.process_thread = ProcessingThread(self)
        self.process_thread.start()

    def update_progress(self, value, max_value):
        self.progress_bar.setMaximum(max_value)
        self.progress_bar.setValue(value)

    def update_status_message(self, message):
        self.status_bar.showMessage(message)

    def processing_completed(self):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.process_button.setEnabled(True)

# Configure the logger
logging.basicConfig(level=logging.ERROR, filename='error_log/error.log',
                    format='%(asctime)s - %(message)s')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ProcessingGUI()
    gui.show()
    sys.exit(app.exec_())
