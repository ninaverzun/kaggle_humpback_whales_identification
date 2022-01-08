Hi!

This script is performs oversampling of the original train.csv to balance between most abundant whales (appearing in ~73 images in the dataset) 
and the rarest ones (appearing only once in the dataset).
Each whale image which is found less than 15 times is copied up to the threshold.

Please make sure that the following prerequisites are fulfilled before running 'whales_oversampling.py':

1) Python 3 should be installed.
2) Python should be part of the PATH environment variable (for windows).
3) The following external libraries are required:
	1. pandas	
	2. numpy
	3. warnings
	4. os
 
4) The script expects "oversampling" folder to be in the same file path as the script, with the following subfile: 
	|
	|_whales_oversampling.py
	|	
	|_oversampling
		|
		|_train.csv
		
5) ALL required resources can be found in "oversampling" folder submitted with the script
