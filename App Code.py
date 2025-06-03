import streamlit as st
import numpy as np
import pandas as pd
import librosa
import os
import joblib
import requests
from googletrans import Translator
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from PyPDF2.errors import PdfReadError
import tempfile
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

if not os.getenv("GOOGLE_API_KEY"):
    st.error("Google API key not loaded!")

# Initialize translator
translator = Translator()

# Configuration
LANGUAGE_DICT = {'HINDI': 'hi-IN', 'MARATHI': 'mr-IN'}
SARVAM_API_KEY = "ff014cc1-c60d-4304-b097-76e135ab6ac1"

# ========== CORE FUNCTIONS ==========

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.warning(f"Couldn't read {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create and save vector store"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create QA chain with prompt template"""
    prompt_template = """
    Answer the question from the context. If answer isn't in context,
    say "answer not available in context". Don't make up answers.
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, 
                          input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def get_answer(question):
    """Get answer from vector store"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, 
                                 allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": question}, 
                        return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        return f"Error: {str(e)}"

def extract_features(audio_file):
    """Extract exactly 38 MFCC features to match the scaler's expectations"""
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract only MFCCs with 38 coefficients
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=38)
    
    # Take mean across time and flatten to 1D array
    features = np.mean(mfccs.T, axis=0)
    
    return features

def speech_to_text(audio_path, language_code):
    """Convert speech to text using Sarvam API and return only the transcript"""
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {'api-subscription-key': SARVAM_API_KEY}
    payload = {
        'model': 'saarika:v1',
        'language_code': LANGUAGE_DICT.get(language_code, 'hi-IN'),
        'with_timesteps': 'false'
    }
    
    try:
        with open(audio_path, 'rb') as f:
            files = [('file', (os.path.basename(audio_path), f, 'audio/wav'))]
            response = requests.post(url, headers=headers, data=payload, files=files)
            
            if response.status_code == 200:
                response_data = response.json()
                transcript = response_data.get("transcript", "")
                # Clean and format the question
                question = transcript.strip()
                if not question.endswith('?'):
                    question += '?'
                return question
            st.error(f"API returned status code {response.status_code}")
            return None
    except Exception as e:
        st.error(f"API request failed: {str(e)}")
        return None

@st.cache_resource
def load_language_models():
    """Load all language detection models"""
    try:
        return {
            'scaler': joblib.load(r"D:\Nidhi\Audio chatbot\scaler.pkl"),
            'encoder': joblib.load(r"D:\Nidhi\Audio chatbot\label_encoder.pkl"),
            'ensemble': joblib.load(r"D:\Nidhi\Audio chatbot\ensemble_language_predictor.pkl"),
            'models': {
                'mha': load_model(r"D:\Nidhi\Audio chatbot\MHA.h5"),
                'bilstm': load_model(r"D:\Nidhi\Audio chatbot\MHA Bilstm.h5"),
                'cnn': load_model(r"D:\Nidhi\Audio chatbot\Final_CNN_BiLSTM_MHA_Model.h5")
            }
        }
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def detect_language(features, models_dict):
    try:
        # Ensure we have exactly 38 features
        if len(features) != 38:
            raise ValueError(f"Expected 38 features, got {len(features)}")
            
        # Reshape for scaler (n_samples=1, n_features=38)
        features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = models_dict['scaler'].transform(features)
        
        # Prepare inputs for each model
        X_mha = features_scaled.reshape(1, 1, 38)  # (batch, timesteps, features)
        X_cnn = features_scaled.reshape(1, 38, 1)   # (batch, features, channels)
        
        # Get predictions
        pred_mha = models_dict['models']['mha'].predict(X_mha, verbose=0)
        pred_bilstm = models_dict['models']['bilstm'].predict(X_mha, verbose=0)
        pred_cnn = models_dict['models']['cnn'].predict(X_cnn, verbose=0)
        
        # Combine predictions
        ensemble_input = np.concatenate([pred_mha, pred_bilstm, pred_cnn], axis=1)
        final_pred = models_dict['ensemble'].predict(ensemble_input)
        
        return models_dict['encoder'].inverse_transform(final_pred)[0]
        
    except Exception as e:
        st.error(f"Detection Error: {str(e)}")
        st.error(f"Input shape: {features.shape if 'features' in locals() else 'N/A'}")
        return None

# ========== STREAMLIT APP ==========
def main():
    st.set_page_config("Multilingual Audio Q&A System", layout="wide")
    st.title("🎙️ Multilingual Audio Q&A System")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_language" not in st.session_state:
        st.session_state.current_language = None
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False
    if "last_audio_file" not in st.session_state:
        st.session_state.last_audio_file = None
    
    models_dict = load_language_models()
    
    # Sidebar
    with st.sidebar:
        st.header("📂 Knowledge Base")
        uploaded_files = st.file_uploader("Upload PDFs", 
                                        accept_multiple_files=True,
                                        type=["pdf"])
        if uploaded_files:
            with st.spinner("Processing documents..."):
                raw_text = get_pdf_text(uploaded_files)
                if raw_text:
                    get_vector_store(get_text_chunks(raw_text))
                    st.success("Knowledge base ready!")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🎤 Ask Question")
        audio_file = st.file_uploader("Upload WAV", type=["wav"], key="audio_uploader")
        
        # Show detected language permanently if we have one
        if st.session_state.current_language:
            st.success(f"Detected language: {st.session_state.current_language}")
        
        if audio_file and models_dict and (audio_file != st.session_state.last_audio_file or not st.session_state.processing_done):
            st.session_state.last_audio_file = audio_file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                audio_path = tmp.name
            
            with st.spinner("Analyzing audio..."):
                try:
                    # Step 1: Feature extraction and language detection
                    features = extract_features(audio_path)
                    language = detect_language(features, models_dict)
                    
                    if language:
                        st.session_state.current_language = language
                        st.success(f"Detected language: {language}")
                        
                        # Step 2: Speech-to-text conversion
                        question = speech_to_text(audio_path, language)
                        if question:
                            st.markdown(f"**🎤 Converted text:** {question}")
                            
                            # Store question
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": question,
                                "language": language
                            })
                            
                            # Step 3: Get answer from RAG
                            with st.spinner("Generating answer..."):
                                answer = get_answer(question)
                                
                                # Step 4: Translate if needed
                                if language != "ENGLISH":
                                    try:
                                        answer = translator.translate(answer, src='en', dest=language.lower()).text
                                    except:
                                        answer += " (translation failed)"
                                
                                # Store answer
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "language": language
                                })
                                
                                st.session_state.processing_done = True
                                st.rerun()
                        else:
                            st.error("Failed to convert speech to text")
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                finally:
                    os.unlink(audio_path)
    
    with col2:
        st.header("💬 Conversation")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant":
                    st.caption(f"Translated to {msg['language']}")
        
        # Reset processing flag when new audio is uploaded
        if audio_file is None:
            st.session_state.processing_done = False
            st.session_state.last_audio_file = None
        elif audio_file != st.session_state.last_audio_file:
            st.session_state.processing_done = False

if __name__ == "__main__":
    main()