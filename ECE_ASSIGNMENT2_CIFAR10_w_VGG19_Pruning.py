# Name: Henry Hoang
# Date: 10/5/25
# Course: ECE 538
# Assignment 2 - Train Image dataset "CIFAR-10" and evaluate VGG19 using TensorFlow

# Disable oneDNN optimizations to avoid compatibility issues with TF-MOT pruning
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

########## IMPORT REQUIRED PACKAGES ##########
# Core libraries for computation, modeling, and visualization
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################
# Configuration block for training and pruning parameters
##############################################################################
mixed_precision.set_global_policy('float32')   # Use float32 to avoid mixed precision issues during pruning
BATCH_SIZE = 32                                # Number of samples per training batch
EPOCHS = 15                                     # Epochs for training unpruned model
FINE_TUNE_EPOCHS = 15                           # Epochs for fine-tuning pruned model
FINAL_SPARSITY = 0.4                            # Target sparsity level for pruning
NUM_CLASSES = 10                                # CIFAR-10 has 10 image classes
PLOT_DIR = "plots"                              # Directory to save plots
os.makedirs(PLOT_DIR, exist_ok=True)            # Create plot directory if it doesn't exist

##############################################################################
# Data loading and preprocessing
##############################################################################
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize image pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Resize images to 224x224 to match VGG19 input requirements
x_train = tf.image.resize(x_train, (224, 224)).numpy()
x_test  = tf.image.resize(x_test, (224, 224)).numpy()

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test  = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# Split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Create TensorFlow datasets with performance optimizations
AUTOTUNE = tf.data.AUTOTUNE
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
val_data   = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
test_data  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

##############################################################################
# Model builder function: constructs VGG19-based classifier
##############################################################################
def build_model(num_classes=NUM_CLASSES):
    inputs = Input(shape=(224, 224, 3), name="input_image")  # Define input shape
    backbone = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)  # Load pretrained VGG19 without top layers
    backbone.trainable = False  # Freeze convolutional base for transfer learning
    x = GlobalAveragePooling2D(name="gap")(backbone.output)  # Reduce spatial dimensions
    x = Dense(256, activation="relu", dtype="float32", name="fc1")(x)  # Fully connected layer
    outputs = Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)  # Output layer
    return Model(inputs=inputs, outputs=outputs, name="vgg19_unpruned")

##############################################################################
# Manual magnitude-based pruning function
##############################################################################
def magnitude_prune_model(model, final_sparsity):
    total_params = 0
    pruned_params = 0
    print("\n--- Layer-wise Pruning Report ---")
    
    # Iterate through layers and prune weights based on magnitude
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            weights = layer.get_weights()
            if not weights: 
                continue
            kernel = weights[0]
            flat = np.abs(kernel).reshape(-1)
            n_total = flat.size
            k = int(np.floor(final_sparsity * n_total))
            if k <= 0: 
                continue
            
            # Determine pruning threshold and apply mask
            thresh = np.partition(flat, k)[k]
            mask = np.abs(kernel) > thresh
            kernel_pruned = kernel * mask

            # Set pruned weights back to the layer
            new_weights = [kernel_pruned]
            if len(weights) > 1:
                new_weights.append(weights[1])  # Bias term
            try:
                layer.set_weights(new_weights)
            except Exception as ex:
                print(f"Warning: could not set weights for {layer.name}: {ex}")

            # Track pruning statistics
            layer_pruned = (n_total - mask.sum())
            total_params += n_total
            pruned_params += layer_pruned
            print(f"{layer.name:<20s} | size={n_total:8d} | pruned={layer_pruned:8d} "
                  f"| sparsity={(100.0*layer_pruned/n_total):5.2f}%")

    global_sparsity = 100.0 * pruned_params / total_params
    print(f"\nManual pruning applied. Global sparsity: {global_sparsity:.2f}% "
          f"({pruned_params}/{total_params} params pruned)")
    return model

##############################################################################
# Utility to compute overall model sparsity
##############################################################################
def compute_model_sparsity(model, prefix=""):
    total = 0
    zeros = 0
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            weights = layer.get_weights()
            if not weights: continue
            kernel = weights[0]
            total += kernel.size
            zeros += np.sum(kernel == 0)
    sparsity = 100.0 * zeros / total if total > 0 else 0.0
    print(f"{prefix} Model Sparsity: {sparsity:.2f}% "
          f"({zeros}/{total} weights are zero)")
    return sparsity

##############################################################################
# Evaluation function: computes metrics and plots confusion matrix
##############################################################################
def evaluate_and_report(model, test_data, y_test, prefix="unpruned"):
    start_time = time.time()
    y_pred_probs = model.predict(test_data)
    end_time = time.time()

    # Convert predictions and labels to class indices
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1).reshape(-1)

    # Compute classification metrics
    acc  = accuracy_score(y_true_classes, y_pred_classes)
    prec = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    rec  = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    f1   = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)

    # Compute hardware metrics
    total_time = end_time - start_time
    num_samples = y_test.shape[0]
    throughput = num_samples / total_time if total_time > 0 else float('inf')
    latency = (total_time / num_samples) * 1000 if num_samples > 0 else 0.0

    # Print metrics
    print(f"\n==== {prefix.upper()} MODEL METRICS ====")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print(f"Total Inference Time: {total_time:.2f} sec | Throughput: {throughput:.2f} samples/sec | Latency: {latency:.2f} ms")

    # Print per-class metrics
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    print("\nPer-Class Metrics:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0))

    # Plot confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({prefix})')
    plt.savefig(os.path.join(PLOT_DIR, f'confusion_matrix_{prefix}.png'))
    plt