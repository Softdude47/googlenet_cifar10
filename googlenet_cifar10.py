import os
import argparse
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# local import
from mas_lib.nn.conv.minigooglenet import MiniGoogLeNet
from mas_lib.callbacks.trainingmonitor import TrainingMonitor

# constants
BATCH_SIZE = 32
NUM_EPOCHS = 70
POWER = 1
LEARNING_RATE = 0.01

# commandline arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="./googlenet_cifar10.h5", help="path to output model")
ap.add_argument("-o", "--output", required=True, help="path output model logs, plot")
args = vars(ap.parse_args())
plot_path = os.path.sep.join([args["output"], f"{os.getpid()}.png"])
json_path = os.path.sep.join([args["output"], f"{os.getpid()}.json"])

# model training callback
@LearningRateScheduler
def poly_decay(current_epoch):
    lr = LEARNING_RATE * ((1 - current_epoch/NUM_EPOCHS) ** POWER)
    return lr

# dataset
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float")
x_test = x_test.astype("float")

# means subtraction
mean = np.mean(x_train, axis=0)
x_train -= mean
x_test -= mean

# encode dataset labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# initialize dataset generator alongside model optimizer
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=30,
                        shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
opt = SGD(learning_rate=0.01, momentum=0.9)

# build and train model
model = MiniGoogLeNet.build(32, 32, 3, 10)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    aug.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(x_train)//BATCH_SIZE,
    validation_data=(x_test, y_test),
    epochs=NUM_EPOCHS,
    callbacks=[poly_decay, TrainingMonitor(plot_path, json_path)]
)

# evaluate model
pred = model.predict(x_test, batch_size=BATCH_SIZE)
pred = np.argmax(pred, axis=1)
report = classification_report(y_true=y_test.argmax(axis=1), y_pred=pred, target_names=class_names)
print(report)

# save model
model.save(args["model"])