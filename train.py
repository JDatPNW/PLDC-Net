import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

import PLDC_Net

pldc_net = PLDC_Net.pldc_net


save_name = "pldc_net"

dataset_dir_name = "PLDC_80"

data_directory = f"./data/{dataset_dir_name}/train"
test_directory = f"./data/{dataset_dir_name}/test/"

data_directory = "./data/benchmark_augmented_all_cutoff/train"
test_directory = './data/benchmark_augmented_all_cutoff/test/'

results_directory = f"./results/benchmark_augmented_all_cutoff/{save_name}/"

if not os.path.exists(results_directory):
    os.makedirs(results_directory)

batch_size = 128
epochs = 150
image_size=(224, 224)
input_size=(224, 224, 3)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    label_mode='categorical', 
    validation_split=0.2,
    subset="training",
    seed=1337,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="validation",
    label_mode='categorical',
    seed=1337,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_directory,
    label_mode='categorical', 
    seed=1337,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    results_directory + save_name + "_best.h5",  
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

optimizer = Adam()


pldc_net.compile(optimizer=optimizer, metrics=["accuracy"], loss=tf.keras.losses.CategoricalCrossentropy())
history = pldc_net.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint_cb]
)

best_model = tf.keras.models.load_model(results_directory + save_name + "_best.h5")

train_loss, train_acc = best_model.evaluate(train_ds)
print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Train Loss: {train_loss:.4f}")

val_loss, val_acc = best_model.evaluate(val_ds)
print(f"\nValidation Accuracy: {val_acc:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

test_loss, test_acc = best_model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

with open(results_directory + save_name + ".txt", "w") as file:
    file.write(f"Train Accuracy: {train_acc:.4f}\n")
    file.write(f"Train Loss: {train_loss:.4f}\n")
    file.write(f"Validation Accuracy: {val_acc:.4f}\n")
    file.write(f"Validation Loss: {val_loss:.4f}\n")
    file.write(f"Test Accuracy: {test_acc:.4f}\n")
    file.write(f"Test Loss: {test_loss:.4f}\n")
    file.write(f"Best Epoch based on Validation Accuracy and epoch used to generate these reults: {np.argmax(history.history['val_accuracy']) + 1}")


pldc_net.save(results_directory + save_name + ".h5")

with open(results_directory + save_name + ".pkl", "wb") as f:
    pickle.dump(history.history, f)

history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
epochs_range = range(len(train_loss))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.axhline(y=test_loss, color='r', linestyle='--', label="Test Loss")  # Add test loss as a horizontal line
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation vs Test Loss")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.axhline(y=test_acc, color='r', linestyle='--', label="Test Accuracy")  # Add test accuracy as a horizontal line
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation vs Test Accuracy")

plt.tight_layout()
plt.savefig(results_directory + save_name + ".png", dpi=300, bbox_inches="tight")  # Higher DPI for better quality
plt.show()
