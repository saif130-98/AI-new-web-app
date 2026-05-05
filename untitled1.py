# ==========================================
# IMPORTS
# ==========================================
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK Tools
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Gensim (Word Embeddings)
from gensim.models import Word2Vec

# Keras / TensorFlow (Deep Learning)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping # Added Early Stopping
import pickle

# Sklearn Metrics
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. LOAD THE TRAINING DATA
# ==========================================
print("Loading data...")
url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
df = pd.read_csv(url, header=None, names=['Topic', 'Title', 'Description'])

cols = [col for col in df.columns if col != 'Topic'] + ['Topic']
df = df[cols]
df['Full_Text'] = df['Title'] + " " + df['Description']

# ==========================================
# 2. DEFINE CLEANING & LEMMATIZATION
# ==========================================
print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() 
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'[^\w\s]|_', '', text) 
    
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    
    cleaned_words = []
    for word, tag in pos_tags:
        if word not in stop_words:
            wn_tag = get_wordnet_pos(tag)
            lemmatized_word = lemmatizer.lemmatize(word, pos=wn_tag)
            cleaned_words.append(lemmatized_word)
            
    return ' '.join(cleaned_words)

# ==========================================
# 3. CREATE CLEANED DATAFRAME
# ==========================================
print("Building the cleaned dataset (this may take a few minutes for the full dataset)...")
df_cleaned = pd.DataFrame()
df_cleaned['Clean_Text'] = df['Full_Text'].apply(clean_text)
df_cleaned['Topic'] = df['Topic']

# ==========================================
# 4. TRAIN WORD EMBEDDINGS (Word2Vec)
# ==========================================
print("\nPreparing text for Embeddings...")
tokenized_sentences = df_cleaned['Clean_Text'].apply(lambda x: x.split()).tolist()

print("Training Word2Vec model...")
w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=2, workers=4)
print(f"Total Unique Embedded Words: {len(w2v_model.wv)}") 

# ==========================================
# 5. PREPARE DATA FOR KERAS (Padding & Target)
# ==========================================
print("\nPreparing sequences for LSTM...")
MAX_WORDS = len(w2v_model.wv) 
MAX_SEQUENCE_LENGTH = 100 

keras_tokenizer = Tokenizer(num_words=MAX_WORDS)
keras_tokenizer.fit_on_texts(df_cleaned['Clean_Text'])

X_sequences = keras_tokenizer.texts_to_sequences(df_cleaned['Clean_Text'])
X = pad_sequences(X_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(f"Shape of Input Tensor (X): {X.shape}")

Y = pd.get_dummies(df_cleaned['Topic']).values
print(f"Shape of Label Tensor (Y): {Y.shape}")

# ==========================================
# 6. BUILD EMBEDDING MATRIX
# ==========================================
print("\nBuilding Embedding Matrix from Word2Vec...")
word_index = keras_tokenizer.word_index
EMBEDDING_DIM = 100 

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print("Embedding Matrix successfully created!")

# ==========================================
# 7. DEFINE & COMPILE THE LSTM (Now as a function)
# ==========================================
print("\nBuilding LSTM Model Architecture...")

def create_lstm_model(vocab_size, embedding_dim, emb_matrix, max_length):
    """Creates and compiles the LSTM Neural Network."""
    lstm_model = Sequential()
    lstm_model.add(Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_dim, 
        weights=[emb_matrix], 
        input_length=max_length, 
        trainable=False 
    ))
    lstm_model.add(SpatialDropout1D(0.2))
    lstm_model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
    lstm_model.add(Dense(4, activation='softmax')) 

    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return lstm_model

# Instantiate the model using the function
vocab_size = len(word_index) + 1
model = create_lstm_model(vocab_size, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH)
print(model.summary())

# ==========================================
# 8. TRAIN THE MODEL (With Early Stopping)
# ==========================================
print("\nTraining the model...")

# Define Early Stopping
# This monitors validation loss, waits for 5 epochs without improvement, then stops and restores the best weights
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True,
    verbose=1
)

# Increased epochs since Early Stopping will manage the cut-off
history = model.fit(
    X, Y, 
    epochs=16, 
    batch_size=64, 
    validation_split=0.2,
    callbacks=[early_stop] # Pass the callback here
)

# ==========================================
# 9. LOAD & PREPARE TEST DATA
# ==========================================
print("\n--- Loading & Cleaning Test Data ---")
test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
df_test = pd.read_csv(test_url, header=None, names=['Topic', 'Title', 'Description'])
df_test['Full_Text'] = df_test['Title'] + " " + df_test['Description']

df_test_cleaned = pd.DataFrame()
df_test_cleaned['Clean_Text'] = df_test['Full_Text'].apply(clean_text)
df_test_cleaned['Topic'] = df_test['Topic']

# ==========================================
# 10. TOKENIZE TEST DATA
# ==========================================
X_test_seq = keras_tokenizer.texts_to_sequences(df_test_cleaned['Clean_Text'])
X_test = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)
Y_test_true_labels = df_test_cleaned['Topic'].values - 1 
Y_test_encoded = pd.get_dummies(df_test_cleaned['Topic']).values

# ==========================================
# 11. EVALUATE MULTI-CLASS PERFORMANCE
# ==========================================
print("\n--- Generating Predictions ---")
predictions = model.predict(X_test)
Y_pred_labels = np.argmax(predictions, axis=1)

print("\n=== CLASSIFICATION REPORT ===")
target_names = ['World (1)', 'Sports (2)', 'Business (3)', 'Sci/Tech (4)']
print(classification_report(Y_test_true_labels, Y_pred_labels, target_names=target_names))

# ==========================================
# 12. PLOTTING PERFORMANCE
# ==========================================
print("\nGenerating Performance Plots...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Accuracy over Epochs
axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Plot 2: Loss over Epochs
axes[1].plot(history.history['loss'], label='Train Loss', marker='o', color='red')
axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='o', color='orange')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

# Plot 3: Confusion Matrix
cm = confusion_matrix(Y_test_true_labels, Y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2], xticklabels=target_names, yticklabels=target_names)
axes[2].set_title('Confusion Matrix on Test Data')
axes[2].set_ylabel('Actual Label')
axes[2].set_xlabel('Predicted Label')
 
plt.tight_layout()
plt.show()

# ==========================================
# 13. SAVE THE MODEL & ASSETS
# ==========================================
print("\n--- Saving Model and Assets ---")
model.save("ag_news_lstm_model2.keras")
print("1. LSTM Model saved as 'ag_news_lstm_model2.keras'")
w2v_model.save("ag_news_word2vec2.model")
print("2. Word2Vec Model saved as 'ag_news_word2vec2.model'")
with open('ag_news_tokenizer2.pkl', 'wb') as handle:
    pickle.dump(keras_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("3. Tokenizer saved as 'ag_news_tokenizer2.pkl'")
print("\nAll assets successfully saved! Your pipeline is ready for production.")