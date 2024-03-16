import tensorflow as tf
from tensorflow import keras
import numpy as np
import os, pathlib, shutil, random

def make_train_folders(train_files, base_dir):
    if "train_tf" not in os.listdir(base_dir):
        os.mkdir(os.path.join(base_dir, "train_tf"))
        for fileList in train_files:
            author = fileList[0].split("/")[-1].split("_")[0]
            if author not in os.listdir(base_dir + "/train_tf"):
                os.mkdir(base_dir + "/train_tf/" + author)
            
            file_counter = 0
            sentences = []
            for filePath in fileList:
                
                # Resulted in unicode decode error due to fancy quotes without latin-1 encoding
                with open(filePath, "r", encoding='latin-1') as f:
                    entire_novel = f.read()
                    sentences = entire_novel.split(".")

                i = 0
                while i < len(sentences):
                    if len(sentences[i].split()) < 40 and i != len(sentences) - 1:
                        sentences[i] += sentences[i+1]
                        sentences.pop(i+1)
                    i += 1
                
            random.shuffle(sentences)
            
            for sentence in sentences:
                with open(base_dir + "/train_tf/" + author + "/" + author + "_" + str(file_counter) + ".txt", "w") as f:
                    f.write(sentence)
                    file_counter += 1
                
                if file_counter == 2000:
                    break


def learn_model(train_files):
    num_classes = len(train_files)
    base_dir = "dataset"
    make_train_folders(train_files, base_dir)
    training_set = keras.utils.text_dataset_from_directory("dataset/train_tf", batch_size=32)
    training_list = list(training_set.as_numpy_iterator())

    print("Training set size: ", len(training_list))

    text_vectorization = tf.keras.layers.TextVectorization(ngrams=3, output_mode="multi_hot", max_tokens=20000, standardize="lower_and_strip_punctuation")
    text_only_training_set = training_set.map(lambda x, y: x)

    text_vectorization.adapt(text_only_training_set)

    binary_2gram_training_ds = training_set.map(lambda x, y: (text_vectorization(x), y))

    max_tokens = 20000

    model = keras.Sequential([
        # text_vectorization,
        keras.Input(shape = (max_tokens,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(50, activation="tanh"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="SparseCategoricalCrossentropy", metrics=["accuracy"])


    model.fit(binary_2gram_training_ds, epochs=10)

    new_model = keras.Sequential([text_vectorization, model])

    new_model.compile(optimizer="adam", loss="SparseCategoricalCrossentropy", metrics=["accuracy"])

    return new_model
