Hi!

This script is cutting whales images according to given bounding boxes saved in CSV format.
Please make sure that the following prerequisites are fulfilled before running 'Whales_cut_bbox.py':

1) Python 3 should be installed.
2) Python should be part of the PATH environment variable (for windows).
3) The following external libraries are required:
	1. pandas	
	2. keras
4) The following file structure is expected: 
	|
	|_Whales_cut_bbox.py
	|	
	|_input
		|
		|_humpback-whale-identification
			|
			|_train
			|_test
			|_bounding_boxes.csv
			|_cropped_train
			|_cropped_test
5) ALL required resources can be found in "input" folder available for download from here:
https://drive.google.com/open?id=1V0_W6hr8k_OHD0JwQDbWDbz2PYBH02hi

6) The original (not cropped) images from the whales identification dataset are expected to be in 'train' and 'test' folders, in the following file locations
	|_Whales_cut_bbox.py
	|	
	|_input
		|
		|_humpback-whale-identification
			|
			|_train
			|_test

7) The csv file containing the bouding boxes corners is expected to be in 'input/humpback-whale-identification' folder, in the following file location:
	|_Whales_cut_bbox.py
	|	
	|_input
		|
		|_humpback-whale-identification
			|
			|_bounding_boxes.csv


