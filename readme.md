# Las file processing Readme
### How to use the Las file processing tool

##  Installation
- git clone this repository or download the zip file and extract it to a folder
- Current release was tested on python 3.10 
- Install python 3.10 from https://www.python.org/downloads/ for windows
- Once python is installed, open command prompt and navigate to the folder where the repository is cloned or extracted
- Run the command in cmd `pip install -r requirements.txt` this will install all the required packages and dependencies
- You are now ready to use the tool

##  Usage

- Add raw files to raw_las folder, make sure the files are in .las format
- Add alias xml files to alias_xml folder, make sure the xml file name is curve_alias.xml
- Run the Las file processing.bat file and wait for the GUI to open, Do not close the terminal cmd window minimize it
- Once the Gui is open select the Source folder, the folder where the raw files are located
- Select the Destination folder, the folder where the processed files will be saved
- Check the apply preconditioning box if you want to apply preconditioning
- Set the drho_matrix value, this is the value that will be used to calculate the density matrix
- Set the n value for Lfilter smoothing curve
- Check the apply fluid properties box if you want to apply fluid properties
- Check the apply multimineral box if you want to apply multimineral, make sure to check fluid properties with this
- Check the apply Electrofacies box if you want to apply Electrofacies
- Set the n_clusters value for number of clusters for Electrofacies
- Check the curves for electrofacies clustering
- Click the Process button and wait for the process to finish
- Once the process is finished, the processed las files will be saved in the destination folder 

## Troubleshooting
- If the process fails, check the error logs
- You can find error logs in the error_log folder if there are any errors in the process