import os
import re
import nltk
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download necessary NLTK data on startup
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Initialize FastAPI
app = FastAPI(title="News Classifier API")

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and utilities
loaded_model = None
loaded_tokenizer = None
stop_words = None
lemmatizer = None
MAX_SEQUENCE_LENGTH = 100

category_names = {
    0: "🌍 World",
    1: "⚽ Sports",
    2: "📈 Business",
    3: "🔬 Sci/Tech"
}

@app.on_event("startup")
async def load_assets():
    global loaded_model, loaded_tokenizer, stop_words, lemmatizer
    
    # Path handling: models are in the parent directory of 'backend'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "ag_news_lstm_model.keras")
    tokenizer_path = os.path.join(base_dir, "ag_news_tokenizer.pkl")
    
    try:
        loaded_model = load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            loaded_tokenizer = pickle.load(handle)
    except Exception as e:
        print(f"Error loading models: {e}")
        
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

class PredictionRequest(BaseModel):
    text: str

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

def clean_text(text: str) -> str:
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
            cleaned_words.append(lemmatizer.lemmatize(word, pos=wn_tag))
            
    return ' '.join(cleaned_words)

@app.post("/api/predict")
async def predict_category(request: PredictionRequest):
    if not loaded_model or not loaded_tokenizer:
        raise HTTPException(status_code=500, detail="Models are not loaded.")
        
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    # Clean text
    cleaned_input = clean_text(text)
    
    # Tokenize and pad
    sequence = loaded_tokenizer.texts_to_sequences([cleaned_input])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Run inference
    prediction_probs = loaded_model.predict(padded_sequence, verbose=0)[0]
    winning_index = int(np.argmax(prediction_probs))
    winning_category = category_names[winning_index]
    
    # Format confidence breakdown
    confidence_breakdown = {}
    for idx, prob in enumerate(prediction_probs):
        confidence_breakdown[category_names[idx]] = float(prob)
        
    # Clean sequence for the "Under the Hood" feature (remove trailing zeros)
    clean_sequence = [int(num) for num in padded_sequence[0] if num != 0]

    return {
        "winning_category": winning_category,
        "confidence_breakdown": confidence_breakdown,
        "cleaned_string": cleaned_input,
        "padded_sequence": clean_sequence
    }
