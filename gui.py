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

            self.parent.update_progress(index + 1, len(las_files))

            # Update the status bar to show progress like "1/4 files"
            self.parent.update_status_message(f"Processing {index}/{len(las_files)} files")

        if electrofacies:
            logs = [pet.Log(x) for x in processed_files]

            # Include the electrofacies processing step in the progress bar
            self.parent.update_progress(len(las_files) + 1, len(las_files) + 1)

            combined_logs = ef.electrofacies(logs=logs,)

            for i, log in enumerate(combined_logs):
                log.write(processed_files[i])  # Overwrite the processed file with electrofacies results

            # Update the status bar for electrofacies step
            self.parent.update_status_message(f"Processing Electrofacies (Step {len(las_files) + 1}/{len(las_files) + 1})")

        self.parent.processing_completed()
        # Update the status bar to show completion status
        self.parent.update_status_message("Processing completed!")

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
        self.setGeometry(200, 200, 400, 300)

        # Create and set up the status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Create and set up the progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addWidget(self.progress_bar)

    def browseSourceFolder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Source Folder')
        self.source_folder_input.setText(folder)

    def browseDestFolder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
        self.dest_folder_input.setText(folder)

    def processFiles(self):
        self.process_button.setEnabled(False)  # Disable the process button while processing
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(1)  # Set a temporary max value

        # Create an instance of the processing thread and start it
        self.process_thread = ProcessingThread(self)
        self.process_thread.start()

    def update_progress(self, value, max_value):
        self.progress_bar.setMaximum(max_value)
        self.progress_bar.setValue(value)

    def update_status_message(self, message):
        self.status_bar.showMessage(message)

    def processing_completed(self):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.process_button.setEnabled(True)  # Re-enable the process button

# ... (Your existing __main__ block)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ProcessingGUI()
    gui.show()
    sys.exit(app.exec_())
