# digit_recognizer.py

import os
import numpy as np
import cv2
import PIL.ImageGrab
from tkinter import *
import pyttsx3  # <--- Added for voice output
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

MODEL_FILENAME = "digit_model.h5"

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# -------- STEP 1: Train the Model (only if not already trained) --------
def train_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    model.save(MODEL_FILENAME)
    print("✅ Model trained and saved as", MODEL_FILENAME)

# -------- STEP 2: Predict the Drawn Digit --------
def predict_digit():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Capture canvas area
    img = PIL.ImageGrab.grab().crop((x, y, x1, y1))
    img = img.convert('L')  # convert to grayscale
    img = img.resize((28, 28))
    img = np.array(img)
    img = cv2.bitwise_not(img)  # make black digits white
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    result_label.config(text=f"Predicted Digit: {digit}")

    # 🔊 Voice output
    engine.say(f"You wrote the digit {digit}")
    engine.runAndWait()

# -------- STEP 3: Setup Drawing GUI --------
def clear_canvas():
    canvas.delete("all")
    result_label.config(text="Draw a digit and click Predict")

def draw(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill='black')

# Train the model if not already saved
if not os.path.exists(MODEL_FILENAME):
    train_model()

# Load trained model
model = load_model(MODEL_FILENAME)

# Create the GUI
root = Tk()
root.title("Handwritten Digit Recognizer with Voice")

canvas = Canvas(root, width=200, height=200, bg="white")
canvas.pack()
canvas.bind("<B1-Motion>", draw)

Button(root, text="Predict", command=predict_digit).pack(pady=5)
Button(root, text="Clear", command=clear_canvas).pack(pady=5)
result_label = Label(root, text="Draw a digit and click Predict", font=("Arial", 14))
result_label.pack(pady=5)

root.mainloop()
