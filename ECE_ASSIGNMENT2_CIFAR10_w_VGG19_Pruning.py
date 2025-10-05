##########################################################################################
# Name: Henry Hoang
# Date: 10/5/25
# Course: ECE 538
# Assignment 2 - Train Image Dataset "CIFAR-10" and Evaluate VGG19 using TensorFlow
#
##########################################################################################

# Disable OneDNN optimization logs for cleaner output
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

########## IMPORT REQUIRED PACKAGES ##########
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################
# Configuration Block
# Declares the parameters used for model training, pruning, and evaluation.
##############################################################################
mixed_precision.set_global_policy('float32')  # Avoid mixed precision issues with pruning

BATCH_SIZE = 32               # Number of samples per training batch
EPOCHS = 15                   # Training epochs for unpruned model
FINE_TUNE_EPOCHS = 15         # Fine-tuning epochs for pruned model
FINAL_SPARSITY = 0.4          # Target sparsity (fraction of weights set to zero)
NUM_CLASSES = 10              # CIFAR-10 has 10 categories
PLOT_DIR = "plots"            # Directory to save output plots
os.makedirs(PLOT_DIR, exist_ok=True)

##############################################################################
# Load and Preprocess CIFAR-10 Dataset
##############################################################################
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Resize CIFAR-10 images (32x32) to VGG19 input size (224x224)
x_train = tf.image.resize(x_train, (224, 224)).numpy()
x_test  = tf.image.resize(x_test, (224, 224)).numpy()

# One-hot encode class labels
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test  = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# Split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# Create TensorFlow data pipelines
AUTOTUNE = tf.data.AUTOTUNE
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .shuffle(10000).batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
    .batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
    .batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

##############################################################################
# Build Base VGG19 Model (Unpruned)
##############################################################################
def build_model(num_classes=NUM_CLASSES):
    """Builds a VGG19-based classifier for CIFAR-10."""
    inputs = Input(shape=(224, 224, 3), name="input_image")
    backbone = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)
    backbone.trainable = False  # Freeze convolutional base (transfer learning)

    # Add custom classification head
    x = GlobalAveragePooling2D(name="gap")(backbone.output)
    x = Dense(256, activation="relu", dtype="float32", name="fc1")(x)
    outputs = Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    return Model(inputs=inputs, outputs=outputs, name="vgg19_unpruned")

##############################################################################
# Manual Magnitude-Based Pruning Function
##############################################################################
def magnitude_prune_model(model, final_sparsity):
    """
    Prunes weights by setting the smallest magnitude values to zero.
    Applies pruning to Dense and Conv2D layers only.
    """
    total_params = 0
    pruned_params = 0
    print("\n--- Layer-wise Pruning Report ---")

    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            weights = layer.get_weights()
            if not weights:
                continue

            # Compute pruning threshold based on absolute weight magnitude
            kernel = weights[0]
            flat = np.abs(kernel).reshape(-1)
            n_total = flat.size
            k = int(np.floor(final_sparsity * n_total))
            if k <= 0:
                continue
            thresh = np.partition(flat, k)[k]

            # Apply pruning mask
            mask = np.abs(kernel) > thresh
            kernel_pruned = kernel * mask

            # Reassign weights
            new_weights = [kernel_pruned]
            if len(weights) > 1:
                new_weights.append(weights[1])
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
# Compute Model Sparsity Utility
##############################################################################
def compute_model_sparsity(model, prefix=""):
    """Computes and prints the percentage of zero weights in the model."""
    total = 0
    zeros = 0
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            weights = layer.get_weights()
            if not weights:
                continue
            kernel = weights[0]
            total += kernel.size
            zeros += np.sum(kernel == 0)
    sparsity = 100.0 * zeros / total if total > 0 else 0.0
    print(f"{prefix} Model Sparsity: {sparsity:.2f}% "
          f"({zeros}/{total} weights are zero)")
    return sparsity

##############################################################################
# Evaluation and Reporting Function
##############################################################################
def evaluate_and_report(model, test_data, y_test, prefix="unpruned"):
    """Evaluates the model and prints classification metrics and inference stats."""
    start_time = time.time()
    y_pred_probs = model.predict(test_data)
    end_time = time.time()

    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1).reshape(-1)

    # Compute metrics
    acc  = accuracy_score(y_true_classes, y_pred_classes)
    prec = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    rec  = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    f1   = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)

    # Compute performance stats
    total_time = end_time - start_time
    num_samples = y_test.shape[0]
    throughput = num_samples / total_time if total_time > 0 else float('inf')
    latency = (total_time / num_samples) * 1000 if num_samples > 0 else 0.0

    # Print summary
    print(f"\n==== {prefix.upper()} MODEL METRICS ====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Inference Time: {total_time:.2f}s | Throughput: {throughput:.2f} samples/s | Latency: {latency:.2f} ms/sample")

    # Generate classification report and confusion matrix
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    print("\nPer-Class Metrics:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({prefix})')
    plt.savefig(os.path.join(PLOT_DIR, f'confusion_matrix_{prefix}.png'))
    plt.close()

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

##############################################################################
# Train Unpruned Model
##############################################################################
model = build_model(NUM_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nTraining Unpruned Model...")
history_unpruned = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Evaluate model before pruning
metrics_unpruned = evaluate_and_report(model, test_data, y_test, prefix="unpruned")
print("\nPRE-pruned Model Summary:")
model.summary()
compute_model_sparsity(model, prefix="Unpruned")

##############################################################################
# Apply Manual Pruning and Fine-Tune
##############################################################################
print("\nApplying manual magnitude pruning...")
model = magnitude_prune_model(model, FINAL_SPARSITY)

# Re-compile with lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nFine-tuning Pruned Model...")
history_pruned = model.fit(train_data, validation_data=val_data, epochs=FINE_TUNE_EPOCHS)

# Evaluate pruned model
metrics_pruned = evaluate_and_report(model, test_data, y_test, prefix="pruned")
print("\nPOST-pruned Model Summary:")
model.summary()
compute_model_sparsity(model, prefix="Pruned")

##############################################################################
# Plot and Compare Training Performance
##############################################################################
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history_unpruned.history['accuracy'], label='Unpruned Train')
plt.plot(history_unpruned.history['val_accuracy'], label='Unpruned Val')
plt.plot([None]*EPOCHS + history_pruned.history['accuracy'], label='Pruned Train')
plt.plot([None]*EPOCHS + history_pruned.history['val_accuracy'], label='Pruned Val')
plt.title('Accuracy Before & After Pruning')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history_unpruned.history['loss'], label='Unpruned Train')
plt.plot(history_unpruned.history['val_loss'], label='Unpruned Val')
plt.plot([None]*EPOCHS + history_pruned.history['loss'], label='Pruned Train')
plt.plot([None]*EPOCHS + history_pruned.history['val_loss'], label='Pruned Val')
plt.title('Loss Before & After Pruning')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'accuracy_loss_compare.png'))
plt.close()
