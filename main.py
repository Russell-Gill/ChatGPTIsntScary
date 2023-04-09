from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.optimizers import SGD
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from dataclasses import dataclass

@dataclass
class ClassEnum:
    HUMAN = 0
    GPT = 1

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
tokenizer.fit_on_texts(train_data_gpt+train_data_human)

human_train_sequences = create_token_sequences(train_data_human, tokenizer, maxlen=max_words)
train_data_gpt = create_token_sequences(train_data_gpt, tokenizer, maxlen=max_words)

labels = np.array([ClassEnum.GPT for _ in range(train_data_gpt.shape[0])])
labels_human = np.array([ClassEnum.HUMAN for _ in range(human_train_sequences.shape[0])])
all_train = np.concatenate((train_data_gpt, human_train_sequences))
all_labels = np.concatenate((labels, labels_human))

model = Sequential([
    layers.Embedding(input_dim=lexical_size, output_dim=64, input_length=max_words),
    layers.Conv1D(15, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.1),
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid',)
])

sgd = SGD(lr=0.01, nesterov=True)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(
    all_train, 
    all_labels, 
    epochs=400, 
    batch_size=32, 
    validation_split=0.2
)

# plot the loss curve for this model 
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.plot(model.history.history['val_accuracy'])
plt.plot(model.history.history['accuracy'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

test_my_input = strip_newlines(get_data("test_data/test_real.txt"))
new_gpt = strip_newlines(get_data("test_data/test_gpt.txt"))

predict_my_input = create_token_sequences(test_my_input, tokenizer, maxlen=max_words)
predict_new_gpt = create_token_sequences(new_gpt, tokenizer, maxlen=max_words)

preds_gpt = [float(x) for x in model.predict(predict_new_gpt)]
preds_human = [float(x) for x in model.predict(predict_my_input)]

print("\n")
print("Average probability of being ChatGPT in human test set:")
print(sum(preds_human)/len(preds_human))
print("Average probability of being ChatGPT in GPT test set:")
print(sum(preds_gpt)/len(preds_gpt))

# create a matplotlib figure with two plots for each of the outputs above as histograms
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(preds_human, bins=20)
ax1.set_title("Human")
ax1.set_xlabel("Probability of being ChatGPT")
ax1.set_ylabel("Result Density")
ax2.hist(preds_gpt, bins=20)
ax2.set_title("GPT")
ax2.set_xlabel("Probability of being ChatGPT")
plt.show()

new_input = ["I wonder if i sound like chatgpt"]
new_input = create_token_sequences(new_input, tokenizer, maxlen=max_words)
print(model.predict(new_input))

new_gpt_input =["I hope my responses are helpful and informative. Let me know if there's anything else I can do for you!"]
new_gpt_input = create_token_sequences(new_gpt_input, tokenizer, maxlen=max_words)
print(model.predict(new_gpt_input))
