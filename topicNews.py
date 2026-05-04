import streamlit as st
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ==========================================
# 1. PAGE SETUP & STYLING
# ==========================================
st.set_page_config(page_title="News Classifier AI", page_icon="📰", layout="wide")

# Custom CSS for a slight UI polish
st.markdown("""
    <style>
    .main-title { font-size: 3rem; font-weight: 800; color: #1E88E5; margin-bottom: 0px;}
    .sub-title { font-size: 1.2rem; color: #555555; margin-bottom: 30px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED ASSETS & SETUP
# ==========================================
@st.cache_resource
def setup_and_load_assets():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    model = load_model("ag_news_lstm_model.keras")
    with open('ag_news_tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return model, tokenizer

with st.spinner("Powering up the Neural Network... 🧠⚡"):
    loaded_model, loaded_tokenizer = setup_and_load_assets()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
MAX_SEQUENCE_LENGTH = 100 

category_names = {
    0: "🌍 World",
    1: "⚽ Sports",
    2: "📈 Business",
    3: "🔬 Sci/Tech"
}

# ==========================================
# 3. TEXT CLEANING FUNCTIONS
# ==========================================
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
            cleaned_words.append(lemmatizer.lemmatize(word, pos=wn_tag))
            
    return ' '.join(cleaned_words)

# ==========================================
# 4. SIDEBAR (Information Panel)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965879.png", width=80) # AI/News Icon
    st.title("About the Engine")
    st.info("This application is powered by a **Long Short-Term Memory (LSTM)** neural network, trained on the AG News dataset using custom Word2Vec embeddings.")
    
    st.write("### Target Categories:")
    for cat in category_names.values():
        st.write(f"- {cat}")
        
    st.divider()
    st.caption("Built with TensorFlow & Streamlit")

# ==========================================
# 5. MAIN APP UI
# ==========================================
st.markdown('<p class="main-title">📰 AI News Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Instantly route news articles to the correct department using Deep Learning.</p>', unsafe_allow_html=True)

# Input Section inside a sleek container
with st.container():
    user_input = st.text_area(
        "Paste an article headline or snippet below:", 
        height=150, 
        placeholder="e.g., Apple announces a new high-speed quantum computer chip that will revolutionize the industry..."
    )
    
    classify_btn = st.button("🔮 Analyze Text", type="primary", use_container_width=True)

# ==========================================
# 6. PREDICTION LOGIC & RESULTS
# ==========================================
if classify_btn:
    if not user_input.strip():
        st.error("⚠️ Please enter some text before analyzing.")
    else:
        with st.spinner("Processing semantics and calculating vectors..."):
            # Process text
            cleaned_input = clean_text(user_input)
            sequence = loaded_tokenizer.texts_to_sequences([cleaned_input])
            padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
            
            # Predict
            prediction_probs = loaded_model.predict(padded_sequence, verbose=0)[0]
            winning_index = np.argmax(prediction_probs)
            winning_category = category_names[winning_index]
            confidence = prediction_probs[winning_index] * 100
            
        st.divider()
        
        # --- TOP RESULT ---
        st.success(f"### The AI classifies this as: **{winning_category}**")
        
        # --- CONFIDENCE BREAKDOWN (2 Columns) ---
        st.write("#### 📊 Neural Network Confidence Breakdown")
        col1, col2 = st.columns(2)
        
        for idx, prob in enumerate(prediction_probs):
            category = category_names[idx]
            percentage = prob * 100
            
            # Put first two in col1, next two in col2
            target_col = col1 if idx < 2 else col2
            
            with target_col:
                st.write(f"**{category}** ({percentage:.2f}%)")
                st.progress(float(prob))

        st.divider()
        
        # --- UNDER THE HOOD (NLP Specific Feature) ---
        with st.expander("🛠️ See Under the Hood: What did the AI actually read?"):
            st.write("Neural networks don't read English the way we do. First, we strip away punctuation, remove filler 'stop words' (like 'the', 'is', 'at'), and reduce words to their base roots (lemmatization).")
            st.write("**Original Text:**")
            st.caption(f"_{user_input}_")
            st.write("**What the AI Saw (Cleaned & Lemmatized):**")
            st.code(cleaned_input)
            st.write("**How it was fed to the Neural Network (Integer Sequence):**")
            # Show the actual sequence array, filtering out the trailing zeros from padding
            clean_sequence = [num for num in padded_sequence[0] if num != 0]
            st.code(str(clean_sequence))