# ECE538_ASSIGNMENT2_HOANG_HENRY
Training and Evaluating VGG19 DNN with CIFAR-10 dataset. Also, apply Pruning comparing before and after applying pruning.

1. I ran these python script on my PC Desktop Windows 10. 
2. They should not error out or freeze.
3. These are some requirement:
	a. Python 3.11
	b. Python package:
		- tensorflow
		- seaborn
		- matplotlib
		- scikit-learn
		- numpy
		- keras-cv-attention-models
		- tf_keras
		-

Please make sure update these package with the latest version if you have any issues.

EXAMPLE COMMAND:
pip install keras-cv-attention-models

pip install --upgrade numpy


4. Please notice: each Epochs can take 10-35mins depending on the the PC. I ran in Cassian average 10min per Epoch. ~2.5hrs per model or 5hrs for pre and post-pruning model.
5. Both models pre and post will run back to back in this same scripts. It will save the images for pre and post confusion matrix and the compared pre/post Loss and accuracy plots.
6. All hardware metrics such as Accuracy, Precision, Recall, F1 score, per-class metrics and model summary will print to terminal.
