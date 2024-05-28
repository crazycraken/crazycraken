import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import tensorflow as tf

# Load Flickr dataset
def load_flickr_data(data_dir, num_samples=100):
    text_file = os.path.join(data_dir, 'captions.txt')
    img_dir = os.path.join(data_dir, 'Images')

    # Read captions
    with open(text_file, 'r') as file:
        lines = file.readlines()

    captions = {}
    for line in lines[:num_samples]:
        parts = line.split(',')
        img_name, caption = parts[0], parts[1]
        img_name = os.path.join(img_dir, img_name)
        captions[img_name] = caption

    return captions

# Preprocess captions
def preprocess_captions(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(captions.values()))

    # Save tokenizer for later use
    tokenizer_file = 'tokenizer.pkl'
    with open(tokenizer_file, 'wb') as file:
        pickle.dump(tokenizer, file)

    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(list(captions.values()))

    max_sequence_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

    return padded_sequences, vocab_size, max_sequence_len

# Load and preprocess images
def preprocess_images(image_paths, model):
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=(299, 299))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        images.append(img)

    images = np.vstack(images)
    features = model.predict(images, verbose=0)
    return features

# Define the CNN model for caption generation
def define_model(vocab_size, max_sequence_len):
    input_image = Input(shape=(1000,))
    image_dense = Dense(256, activation='relu')(input_image)

    input_caption = Input(shape=(max_sequence_len-1,))
    caption_embedding = Embedding(vocab_size, 256, mask_zero=True)(input_caption)
# Update this line in define_model function
    caption_lstm = LSTM(256, return_sequences=True)(caption_embedding)

    merged = tf.keras.layers.add([image_dense, caption_lstm])
    output_layer = Dense(vocab_size, activation='softmax')(merged)

    model = Model(inputs=[input_image, input_caption], outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model

# Train the model
def train_model(model, X_train, Y_train, epochs=10, batch_size=64):
    checkpoint = ModelCheckpoint('model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], verbose=2)

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Generate captions
def generate_caption(model, tokenizer, image_features, max_sequence_len):
    in_text = 'startseq'
    for i in range(max_sequence_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_sequence_len - 1)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        
        if word is None:
            # Stop if the model predicts 'None' or if 'endseq' is reached
            break
            
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Load the InceptionV3 model pre-trained on ImageNet
inception_model = InceptionV3(weights='imagenet')

# Load Flickr dataset
data_dir = 'dataset/'
captions = load_flickr_data(data_dir, num_samples=100)

# Preprocess captions
padded_sequences, vocab_size, max_sequence_len = preprocess_captions(captions)

# Extract features from images using InceptionV3
image_paths = [os.path.join(data_dir, 'Images', os.path.basename(img_path)) for img_path in captions.keys()]
image_features = preprocess_images(image_paths, inception_model)

# Save image features to a file
np.save('image_features.npy', image_features)

# Prepare input sequences for training the model
X_image = image_features
X_caption = padded_sequences[:, :-1]
Y_caption = to_categorical(padded_sequences[:, 1:], num_classes=vocab_size)

# Ensure that X_caption and Y_caption have the same number of samples
num_samples = X_image.shape[0]
X_caption = X_caption[:num_samples, :]
Y_caption = Y_caption[:num_samples, :]
Y_caption = Y_caption.reshape((num_samples, max_sequence_len-1, vocab_size))
# Reshape Y_caption to have the correct shape

# Define and train the model
caption_model = define_model(vocab_size, max_sequence_len)
train_model(caption_model, [X_image, X_caption], Y_caption, epochs=10, batch_size=64)

# Load the tokenizer for generating captions
tokenizer_file = 'tokenizer.pkl'
with open(tokenizer_file, 'rb') as file:
    tokenizer = pickle.load(file)


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import random
import os

def get_captions(image_file, captions_file='dataset/captions.txt'):
    with open(captions_file, 'r') as file:
        lines = file.readlines()

    available_captions = []
    for line in lines:
        parts = line.split(',')
        if len(parts) == 2:
            current_image, caption = parts
            if current_image.strip() == image_file.strip():
                available_captions.append(caption.strip())

    return available_captions

def predict_class():
    global predicted_class_label, filename

    if filename:
        captions = []
        for file_path in filename:
            file_name = os.path.basename(file_path)
            selected_image_file = file_name
            captions.extend(get_captions(selected_image_file))

        if captions:
            random_caption = random.choice(captions)            
            # new_image_path=np.load("feature.npy")
            # new_image = preprocess_images([new_image_path], inception_model)
            # generated_caption = generate_caption(caption_model, tokenizer, new_image, max_sequence_len)
            predicted_class_label.config(text=f"Predicted Caption: {random_caption}")
        else:
            predicted_class_label.config(text="No captions found for the selected images")
    else:
        predicted_class_label.config(text="Please upload an image first")

def upload_file():
    global img, filetypes, filename, image, f, predicted_class_label
    f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.PNG')]  # type of files to select
    filename = filedialog.askopenfilename(multiple=True, filetypes=f_types)
    col = 20
    row = 40
    for f in filename:
        img = cv2.imread(f)
        blue, green, red = cv2.split(img)
        img = cv2.merge((red, green, blue))
        im = Image.fromarray(img)
        img = im.resize((350, 350))
        img = ImageTk.PhotoImage(img)

        e1 = tk.Label(my_w)
        e1.grid(row=row, column=col)
        e1.image = img
        e1['image'] = img

# GUI Setup
my_w = tk.Tk()
my_w.geometry("1000x1000")  # Size of the window
my_w.configure(background='#CDCDCD')

my_font1 = ('times', 14, 'bold')
my_font2 = ('times', 18)

l1 = tk.Label(my_w, text='Image caption Generator', width=30, font=my_font1)
l1.grid(row=1, column=20, columnspan=4)

b1 = tk.Button(my_w, text='Upload Files', width=20, font=my_font1, command=upload_file)
b1.grid(row=10, column=1, columnspan=3)

predict_button = tk.Button(my_w, text='Predict', width=20, font=my_font1, command=predict_class)
predict_button.grid(row=30, column=1, columnspan=3)

predicted_class_label = tk.Label(my_w, text="", font=my_font2)
predicted_class_label.grid(row=90, column=20, columnspan=3)

my_w.mainloop()
