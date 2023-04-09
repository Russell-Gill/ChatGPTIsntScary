from main import create_token_sequences
from tensorflow import keras
model = keras.models.load_model('model.h5')

new_my_input_sequences = create_token_sequences(["You can always mess around with my project that I’m doing if you want to feel extra demoralised"], tokenizer, maxlen=max_words)
print(model.predict(new_my_input_sequences))