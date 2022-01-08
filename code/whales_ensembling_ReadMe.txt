Hi!

This script is averaging the predictions of four best performing models:

1) Resnet50 		(image size: 100*100, optimizer: Adam) 		-  Kaggle LB0.426
2) InceptionResnetV2 	(image size: 128*128, optimizer: Adam) 		-  Kaggle LB0.515
3) Xception 		(image size: 128*128, optimizer: Adam) 		-  Kaggle LB0.449
4) Xception 		(image size: 128*128, optimizer: Adagrad) 	-  Kaggle LB0.423

Best score achieved with the combination of the first 2 in the above list: Kaggle LB0.545

Please make sure that the following prerequisites are fulfilled before running 'whales_ensembling.py':

1) Python 3 should be installed.
2) Python should be part of the PATH environment variable (for windows).
3) The following external libraries are required:
	1. pandas	
	2. numpy
	3. warnings
 
4) The script expects "ensembling" folder to be in the same file path as the script, with the following subfiles: 
	|
	|_whales_ensembling.py
	|	
	|_ensembling
		|
		|_sample_submission.csv
		|_train.csv
		|_filenames.npy
		|_predictions _xception_lb_0_449.npy
		|_predictions_inceptionresnetv2_lb_0_515.npy
		|_predictions_resnet50_lb_0_426.npy
		|_predictions_xception_adagrad_lb_0_423.npy
		
5) ALL required resources can be found in "ensembling" folder submitted with the script
