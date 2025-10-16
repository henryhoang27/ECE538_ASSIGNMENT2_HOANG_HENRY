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
4. Pruned VGG19 CIFAR-10 with Manual Magnitude Pruning and Data Augmentation
This project demonstrates manual weight pruning, fine-tuning, and performance comparison on a VGG19 model trained using the CIFAR-10 dataset.  
The workflow highlights how structured and unstructured pruning affects model performance, accuracy, and sparsity.
5.Overview
This script:
- Trains a VGG19 (ImageNet pre-trained) model on the CIFAR-10 dataset.  
- Applies on-the-fly data augmentation to improve generalization.  
- Performs manual magnitude-based pruning to remove low-importance weights.  
- Fine-tunes the pruned model to recover accuracy.  
- Compares **accuracy, F1-score, latency, throughput**, and **sparsity** before and after pruning.  
- Visualizes results through confusion matrices and performance plots.

6. Base Architecture:
- Model: VGG19 (from `tf.keras.applications`)  
- Pre-trained on: ImageNet  
- Modified for: CIFAR-10 (10-class classification)  
- Added layers:
  -GlobalAveragePooling2D
  -Dense(256, activation="relu")
  -Dense(10, activation="softmax")

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
7. Dataset and Preprocessing:

a. Configuration
| Parameter | Description | Default |
|------------|-------------|----------|
| BATCH_SIZE | Training batch size | 32 |
| EPOCHS | Epochs for initial (unpruned) training | 1 |
| FINE_TUNE_EPOCHS | Epochs for fine-tuning pruned model | 1 |
| FINAL_SPARSITY | Target global sparsity for pruning | 0.4 (40%) |
| NUM_CLASSES | Number of CIFAR-10 classes | 10 |
| PLOT_DIR | Directory to save plots | plots/ |

b. Data Preparation
- Dataset: **CIFAR-10**, 60,000 images in 10 classes  
- Resize from 32×32 to 224×224 for VGG19 input  
- Normalize to [0, 1] and one-hot encode labels  
- Split 80/20 train-validation + 10k test set

c. Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

8. Pruning Method:
	a. Manual Magnitude-Based Pruning - Unstructured pruning based on smallest-magnitude weights.
	b. Mechanism:
		1. Collect layer weights (Conv2D & Dense)
		2. Flatten and sort absolute values
		3. Zero out lowest final_sparsity fraction
		4. Reassign pruned weights to model
	c. Fine-Tuning - After pruning, retrain with a smaller learning rate (`1e-5`) to regain performance.

9. Evaluation and Metrics
	a. Metrics:
	- Accuracy, Precision, Recall, F1-Score
	- Inference latency and throughput
	- Model sparsity before/after pruning
	b. Visualization:
	- Confusion matrices (`confusion_matrix_unpruned.png`, `confusion_matrix_pruned.png`)
	- Accuracy/Loss comparison (`accuracy_loss_compare.png`
10. Outputs and Results
	- All generated plots and metrics are saved under:
		plots/
		├── confusion_matrix_unpruned.png
		├── confusion_matrix_pruned.png
		└── accuracy_loss_compare.png

11. Training Phases
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

12. Key Functions:
	- build_model():Builds and compiles the modified VGG19 model
	- magnitude_prune_model():Manually prunes model weights by magnitud
	- compute_model_sparsity():Calculates total sparsity across pruned layers
	- evaluate_and_report():Evaluates model and generates metrics + plots


