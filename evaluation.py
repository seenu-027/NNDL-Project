# =====================================================
# evaluation.py  (FINAL VERSION)
# Generates:
#   1) confusion_matrices.png  -> train + test side by side
#   2) precision_recall_curve.png
# =====================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, classification_report
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ----------------------------
# CONFIG
# ----------------------------
DATASET_PATH = r"D:/NNDL PROJECT 2/archive/mnist_png"
MODEL_PATH = "bestmodel_png.h5"
IMG_SIZE = (28, 28)
BATCH_SIZE = 128

# ----------------------------
# LOAD MODEL
# ----------------------------
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ----------------------------
# LOAD DATA
# ----------------------------
train_dir = os.path.join(DATASET_PATH, "train")
test_dir  = os.path.join(DATASET_PATH, "test")

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

train_steps = int(np.ceil(train_gen.samples / BATCH_SIZE))
test_steps  = int(np.ceil(test_gen.samples / BATCH_SIZE))

# ----------------------------
# TRAINING PREDICTIONS
# ----------------------------
print("\nPredicting Training Data...")
train_gen.reset()
y_prob_train = model.predict(train_gen, steps=train_steps, verbose=1)
y_pred_train = np.argmax(y_prob_train, axis=1)
y_true_train = train_gen.classes[:len(y_pred_train)]

# ----------------------------
# TEST PREDICTIONS
# ----------------------------
print("\nPredicting Test Data...")
test_gen.reset()
y_prob_test = model.predict(test_gen, steps=test_steps, verbose=1)
y_pred_test = np.argmax(y_prob_test, axis=1)
y_true_test = test_gen.classes[:len(y_pred_test)]

# ----------------------------
# CONFUSION MATRICES
# ----------------------------
cm_train = confusion_matrix(y_true_train, y_pred_train)
cm_test  = confusion_matrix(y_true_test,  y_pred_test)

labels = list(train_gen.class_indices.keys())

# ---- Save Train + Test Confusion Matrix (side by side) ----
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=labels)
disp1.plot(ax=ax[0], cmap='Blues', colorbar=False, values_format='d')
ax[0].set_title("Training Confusion Matrix")

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=labels)
disp2.plot(ax=ax[1], cmap='Greens', colorbar=False, values_format='d')
ax[1].set_title("Testing Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=300)
plt.show()
print("Saved confusion_matrices.png")

# ----------------------------
# PRECISION–RECALL CURVE
# ----------------------------
y_true_test_bin = np.eye(10)[y_true_test]

plt.figure(figsize=(10, 7))
for digit in range(10):
    precision, recall, _ = precision_recall_curve(
        y_true_test_bin[:, digit],
        y_prob_test[:, digit]
    )
    plt.plot(recall, precision, lw=2, label=f"Digit {digit}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Test Set - PNG Model)")
plt.legend()
plt.grid(True)
plt.savefig("precision_recall_curve.png", dpi=300)
plt.show()

print("\nSaved precision_recall_curve.png")
