import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

test_my_input = ["I wonder how much I sound like chat gpt?", "do I sound like chatgpt?", "I swear, i'm totally not a bot"]

def get_data(location: str) -> list:
    lines = []
    with open(location, "r") as f:
        for line in f:
            lines.append(line)
    return lines

def strip_newlines(lines: list) -> list:
    return [line.strip("\n") for line in lines]

def create_token_sequences(lines: list, tokenizer: Tokenizer, maxlen=10) -> list:
    sequences = tokenizer.texts_to_sequences(lines)
    return pad_sequences(sequences, maxlen)

train_data_gpt = strip_newlines(get_data("data/gpt.txt"))
train_data_human = strip_newlines(get_data("data/human.txt"))

lexical_size = 1000
max_words = 10

tokenizer = Tokenizer(num_words=lexical_size)
tokenizer.fit_on_texts(train_data_gpt)

human_train_sequences = create_token_sequences(train_data_human, tokenizer, maxlen=max_words)
train_data_gpt = create_token_sequences(train_data_gpt, tokenizer, maxlen=max_words)

labels = np.array([1 for _ in range(train_data_gpt.shape[0])])
labels_human = np.array([0 for _ in range(human_train_sequences.shape[0])])
all_train = np.concatenate((train_data_gpt, human_train_sequences))
all_labels = np.concatenate((labels, labels_human))

model = Sequential([
    layers.Embedding(input_dim=lexical_size, output_dim=15, input_length=max_words),
    layers.Dense(15, activation='relu'),
    layers.Dense(5, activation='relu'),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(all_train, all_labels, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    
predict_my_input = create_token_sequences(test_my_input, tokenizer, maxlen=max_words)
print(model.predict(predict_my_input))