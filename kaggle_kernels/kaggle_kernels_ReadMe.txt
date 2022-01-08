Hi!

"kaggle_kernels" folder contains the pyton notebooks that are implementing the preprocessing and training.

Below explanation is relevant to most Kernels code included in "kaggle_kernels" folder.

E.g.:
1) "resnet_50_kaggle_lb_0_426\resnet50_kaggle_first_30_epochs.ipynb"				: Resnet50 		(image size: 100*100, optimizer: Adam) -  Kaggle LB0.406
2) "resnet_50_kaggle_lb_0_426\resnet50_kaggle_LB_0_426.ipynb"					: Resnet50 		(image size: 100*100, optimizer: Adam) -  Kaggle LB0.426
3) "inceptionresnetv2_kaggle_lb_0_515\inceptionresnetv2_kaggle_LB_0_515.ipynb"			: InceptionResnetV2 	(image size: 128*128, optimizer: Adam) -  Kaggle LB0.515
4) "xception_kaggle_lb_0_449\xception_kaggle_LB_0_449.ipynb"					: Xception 		(image size: 128*128, optimizer: Adam) -  Kaggle LB0.449
5) "inceptionresnetv2_oversampling\inceptionresnetv2_oversampling_part1_epochs1to20.ipynb" 	: InceptionResnetV2 model(image size: 128*128, optimizer: Adam) 

Code general structure:

1) Train csv loading and presentation of the "head" and "describe" tables
2) Presentation of one random cropped tail image
3) Preprocessing of the train dataframe: filtering out all "new_whale" tagged images
4) Presentation of the "head" of train dataframe without "new_whale" label
5) Collecting all unique labels (without "new_whale")
6) Imports
7) Definition of "prepareImages" method to preprocess images: 
	- loading each image in "RGB" mode
	- resizing to required size (within available RAM limitations)
	- converting ot numpy array
	- applying Keras "preprocess_input" of relevant model (e.g. keras.applications.inception_resnet_v2)
8) Definition of "remove_new_whale" method to preprocess image lables and create a dictionary that maps unique label index to each image file name.
9) Editing the train dataframe and replacing each label with it's integer index according to the mapping in step 8
10) Presentation of the "head" of train dataframe with integer labels
11) Definition of "prepare_labels" method that encodes all train labels using "OneHotEncoder"
12) Applying prepare_labels to the train dataframe
13) Applying prepareImages to all train images (all train images are now loaded to RAM)
14) Normalizing train images by dividing pixels in 255
15) CNN Model definition according to the "transfer learning" example in MAMAN 13 (detailed in Part #2 of my work - "Work Methods")
16) Setting all layers to be trainable
17) Model compilation
18) Presentation of Model summary
19) Performing the model training in a verbose matter so that loss and accuracy are printed for each Epoch
20) Evaluating the model accuracy on a subset of the train set
21) Saving model and weights to disc in H5 formats
22) Plotting train accuracy
23) Plotting train loss
24) Listing the length of the test examples
25) Creating a dataframe to hold the test results
26) Applying prepareImages to all test images (all test images are now loaded to RAM)
27) Model predictions for test images
Alternatively to steps #26 and #27- Creating "ImageDataGenerator" and running Keras "flow_from_dataframe"
to avoid loading all images to RAM.
28) Saving predictions, indices, file names etc. in ".npy" format to allow ensembling / local loading / debugging of predictions
29) Adding 'new_whale' back to the head of the unique labels list 
30) Definition of "add_new_whale_to_predictions" method that adds a column of 'new_whale' prediction probabilities according to selected threshold (elaborated in part 3.2 of "Data Preprocessing")
31) Definition of "create_results_csv" method
32) Envoking "add_new_whale_to_predictions" and finally "create_results_csv" 
33) END
