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
7. Base Architecture:
- Model: VGG19 (from `tf.keras.applications`)  
- Pre-trained on: ImageNet  
- Modified for: CIFAR-10 (10-class classification)  
- Added layers:
  -GlobalAveragePooling2D
  -Dense(256, activation="relu")
  -Dense(10, activation="softmax")
- Trainable layers:  
  Initially, all VGG19 layers are **frozen** during fine-tuning.
- Model Structure Summary:
Input (224x224x3)
│
├── VGG19 (pretrained, frozen)
│
├── GlobalAveragePooling2D
│
├── Dense(256, ReLU)
│
└── Dense(10, Softmax)
- This is not using the data argument because the goal is comparing the before and after pruning and using the data argument will essentially random transformations (flips, crops, rotations), making the training data distribution slightly different every epoch. And in result, the accuracy difference might then come from data variability, not pruning.
8. Dataset and Preprocessing:
a Dataset
- Source: `tf.keras.datasets.cifar10`
- Classes: 10 (`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`)
- Training Samples: 50,000  
- Testing Samples: 10,000
b. Preprocessing Steps
	1. Normalize pixel values to `[0, 1]`.
	2. Resize all images from `(32, 32)` → `(224, 224)` to match VGG19 input size.
	3. One-hot encode class labels.
	4. Split training set into training (80%) and validation (20%) subsets.
	5. Use TensorFlow `tf.data` pipelines with:
	   - Shuffling
	   - Batching
	   - Caching
	   - Prefetching
9. Pruning Method:
- Manual Magnitude-Based Pruning
a.Unlike TensorFlow Model Optimization Toolkit (TF-MOT), this script performs manual static pruning by directly modifying model weights.
b.Mechanism: For each Dense and Conv2D layer:
	1. Flatten the absolute kernel weights.
	2. Determine a pruning threshold such that a fraction `FINAL_SPARSITY` (e.g., 0.4) of smallest-magnitude weights are zeroed.
	3. Apply a binary mask to prune these weights.
	4. Reassign pruned weights to the layer.
c. Formula
threshold = kth smallest(|weights|)
mask = |weights| > threshold
pruned_weights = weights * mask

d. Sparsity Metrics
- Layer-wise sparsity and global sparsity are printed.
e. Evaluation and Metrics
	a. The evaluation is performed before and after pruning using the test set.
	b. Metrics:
		- Accuracy  
		- Precision (weighted)  
		- Recall (weighted)  
		- F1-score (weighted)  
		- Per-class classification report  
		- Inference time, throughput (samples/sec), and latency (ms/sample)
	c. Additional Visualization:
	- Confusion matrices (`confusion_matrix_unpruned.png`, `confusion_matrix_pruned.png`)
	- Accuracy/Loss comparison (`accuracy_loss_compare.png`
f. Training Phases
	Phase 1: Unpruned Training
		- Optimizer: Adam (learning rate = 1e-4)  
		- Epochs: 15  
		- Model saved in memory for pruning
	Phase 2: Manual Pruning
		-Applies 40% weight sparsity (by default `FINAL_SPARSITY = 0.4`)
	Phase 3: Fine-Tuning
		- Optimizer: Adam (learning rate = 1e-5)  
		- Epochs: 15  
		- Allows pruned model to recover accuracy.
e. Outputs and Results
	- All generated plots and metrics are saved under:
		plots/
		├── confusion_matrix_unpruned.png
		├── confusion_matrix_pruned.png
		└── accuracy_loss_compare.png
		
g. Key Functions:
build_model():Builds and compiles the modified VGG19 model
magnitude_prune_model():Manually prunes model weights by magnitud
compute_model_sparsity():Calculates total sparsity across pruned layers
:evaluate_and_report():Evaluates model and generates metrics + plots
