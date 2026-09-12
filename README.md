AI News Classifier: Deep Learning Text Triage
An end-to-end Natural Language Processing (NLP) pipeline and interactive web application designed to automatically categorize unstructured news text. By leveraging custom-trained semantic embeddings and a sequential neural network, this system acts as an automated triage engine, routing text into four distinct categories: World, Sports, Business, and Sci/Tech.

🧠 The Problem & Solution
The Problem: Organizations face a massive information bottleneck when manually reading and routing thousands of unstructured text documents or news articles daily. Manual sorting is slow, expensive, and scales poorly.
The Solution: This project automates the editorial sorting process. By combining a custom Word2Vec embedding model (for semantic understanding) with a Long Short-Term Memory (LSTM) neural network (for sequential context), the system instantly comprehends and categorizes raw text with 91% accuracy. The model is deployed via a Streamlit web interface for real-time user interaction.

🛠️ Technology Stack
Language: Python 3.x

Deep Learning: TensorFlow / Keras (LSTM, SpatialDropout1D, Early Stopping)

NLP & Text Processing: NLTK (WordNetLemmatizer, POS Tagging), Regex

Word Embeddings: Gensim (Word2Vec)

Web Deployment: Streamlit

Data Manipulation & Metrics: Pandas, NumPy, Scikit-learn

🏗️ Project Architecture & Workflow
Smart Preprocessing: Raw text from the AG News dataset is cleaned using Regex to remove numbers, URLs, and punctuation. NLTK is used for Part-of-Speech (POS) tagging and context-aware lemmatization, reducing words to their true dictionary roots rather than blindly stemming them.

Semantic Mapping (Word2Vec): A custom Word2Vec model was trained directly on the corpus to create a 100-dimensional semantic dictionary. This maps words into a mathematical space where contextually similar words cluster together.

LSTM Neural Network: An LSTM network processes the text sequentially. The custom Word2Vec weights are loaded into the first embedding layer and frozen (trainable=False) to preserve the semantic dictionary. The network utilizes recurrent dropout and spatial 1D dropout to prevent overfitting.

Optimized Training: The model training utilizes an Early Stopping callback monitoring validation loss, automatically halting at Epoch 5 to capture the exact mathematical peak of its generalization performance.

Interactive Web App: The saved .keras model and .pkl Tokenizer are deployed via Streamlit. The app utilizes resource caching for instant load times and features an "Under the Hood" module to visually explain the NLP pipeline to users.
