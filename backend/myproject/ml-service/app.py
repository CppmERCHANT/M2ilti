import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)
CORS(app)

print("=" * 50)
print("🔧 LOADING TRAINED MODEL AND DATASET")
print("=" * 50)

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print("📚 Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
print("✅ Tokenizer loaded successfully!")

# Update paths to be relative to this file
model_path = os.path.join(current_dir, "model", "distilbert_model.pth")
dataset_path = os.path.join(current_dir, "dodo.csv")

model_loaded = False
model = None

if os.path.exists(model_path):
    print(f"📁 Found model file: {model_path}")
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📊 File size: {file_size:.2f} MB")
    
    try:
        print("🔄 Loading your trained model...")
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        if isinstance(model, torch.nn.Module):
            print("✅ Successfully loaded your trained model!")
            model_loaded = True
            model.eval()
            
            if hasattr(model, 'num_labels'):
                print(f"📊 Model configured for {model.num_labels} classes")
            
        else:
            print(f"❌ Unexpected model type: {type(model)}")
            model = None
            
    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
        model = None

# Fallback to base model if loading failed
if model is None:
    print("🔄 Loading base pretrained model as fallback...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-multilingual-cased", 
        num_labels=5
    )
    model.eval()
    print("✅ Base model loaded!")

# Load dataset for recommendations
dataset = None
all_formations = []
formation_tfidf_vectorizer = None
formation_tfidf_matrix = None

def clean_formation_text(text):
    """Clean and normalize formation text"""
    if pd.isna(text) or not text:
        return ""
    
    cleaned = re.sub(r'\s+', ' ', str(text).strip())
    cleaned = re.sub(r'^(formation\s+)|(cours\s+)|(module\s+)', '', cleaned, flags=re.IGNORECASE)
    
    return cleaned

def extract_formations_from_dataset(dataset):
    """Extract all individual formations from the dataset"""
    formations = []
    formation_sources = []
    
    for idx, row in dataset.iterrows():
        inped_value = row.get('inped', '')
        if pd.isna(inped_value) or not inped_value:
            continue
            
        individual_formations = [f.strip() for f in str(inped_value).split(';') if f.strip()]
        
        for formation in individual_formations:
            cleaned_formation = clean_formation_text(formation)
            if len(cleaned_formation) > 3:
                formations.append(cleaned_formation)
                formation_sources.append(idx)
    
    return formations, formation_sources

try:
    if os.path.exists(dataset_path):
        print("📊 Loading dataset for recommendations...")
        dataset = pd.read_csv(dataset_path)
        
        if 'inped' in dataset.columns:
            print(f"✅ Dataset loaded! Found {len(dataset)} records with 'inped' column")
            
            print("🔧 Extracting individual formations...")
            all_formations, formation_sources = extract_formations_from_dataset(dataset)
            print(f"✅ Extracted {len(all_formations)} individual formations")
            
            if all_formations:
                print("🔧 Setting up formation recommendation system...")
                formation_tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000, 
                    stop_words='english', 
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.8
                )
                formation_tfidf_matrix = formation_tfidf_vectorizer.fit_transform(all_formations)
                print("✅ Formation recommendation system ready!")
            else:
                print("❌ No valid formations found in dataset")
                
        else:
            print("❌ 'inped' column not found in dataset")
            dataset = None
    else:
        print(f"❌ Dataset file not found: {dataset_path}")
        
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    dataset = None

def predict_sentiment(text):
    """Predict sentiment for given text"""
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = torch.max(probabilities).item()
            all_probs = probabilities[0].tolist()
        
        return {
            'class': predicted_class + 1,
            'confidence': confidence,
            'raw_prediction': predicted_class,
            'all_probabilities': all_probs
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise e

def get_sentiment_label(prediction_class):
    """Convert numerical prediction to sentiment label"""
    sentiment_labels = {
        1: "Very Negative",
        2: "Negative", 
        3: "Neutral",
        4: "Positive",
        5: "Very Positive"
    }
    return sentiment_labels.get(prediction_class, "Unknown")

def get_formation_recommendations(user_text, sentiment_class, num_recommendations=10):
    """Get formation recommendations based on user input"""
    if not all_formations or formation_tfidf_vectorizer is None:
        return []
    
    try:
        cleaned_user_text = clean_formation_text(user_text)
        user_text_lower = user_text.lower().strip()
        
        user_tfidf = formation_tfidf_vectorizer.transform([cleaned_user_text])
        similarities = cosine_similarity(user_tfidf, formation_tfidf_matrix).flatten()
        
        recommendations = []
        seen_formations = set()
        
        keyword_matches = []
        similarity_matches = []
        
        for idx, formation in enumerate(all_formations):
            formation_lower = formation.lower().strip()
            formation_normalized = formation_lower
            
            if formation_normalized in seen_formations:
                continue
                
            similarity_score = similarities[idx]
            
            keyword_relevance = 0
            user_words = user_text_lower.split()
            formation_words = formation_lower.split()
            
            for user_word in user_words:
                if len(user_word) > 2:
                    for formation_word in formation_words:
                        if user_word == formation_word:
                            keyword_relevance += 1.0
                        elif user_word in formation_word or formation_word in user_word:
                            keyword_relevance += 0.7
            
            if keyword_relevance > 0:
                combined_score = (keyword_relevance * 0.7) + (similarity_score * 0.3)
                keyword_matches.append({
                    'formation': formation.strip(),
                    'similarity_score': round(similarity_score, 4),
                    'keyword_relevance': round(keyword_relevance, 4),
                    'relevance_score': round(combined_score, 4),
                })
            elif similarity_score > 0.1:
                similarity_matches.append({
                    'formation': formation.strip(),
                    'similarity_score': round(similarity_score, 4),
                    'keyword_relevance': 0,
                    'relevance_score': round(similarity_score, 4),
                })
            
            seen_formations.add(formation_normalized)
        
        keyword_matches.sort(key=lambda x: x['relevance_score'], reverse=True)
        similarity_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        recommendations = keyword_matches[:num_recommendations]
        
        if len(recommendations) < num_recommendations:
            remaining_slots = num_recommendations - len(recommendations)
            recommendations.extend(similarity_matches[:remaining_slots])
        
        recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return recommendations[:num_recommendations]
        
    except Exception as e:
        print(f"Error generating formation recommendations: {e}")
        return []

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_text = request.form.get('review', '').strip()
        
        if not user_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Make sentiment prediction
        prediction_result = predict_sentiment(user_text)
        sentiment_label = get_sentiment_label(prediction_result['class'])
        
        # Get formation recommendations
        formation_recommendations = get_formation_recommendations(user_text, prediction_result['class'])
        
        # Format recommendations
        formatted_recommendations = []
        for formation in formation_recommendations:
            formatted_recommendations.append({
                'text': formation['formation'],
                'similarity_score': formation['similarity_score'],
                'hybrid_score': formation['relevance_score'],
                'keyword_relevance': formation.get('keyword_relevance', 0),
            })
        
        response = {
            'text': user_text,
            'model_loaded': model_loaded,
            'sentiment': {
                'class': prediction_result['class'],
                'label': sentiment_label,
                'confidence': round(prediction_result['confidence'], 3)
            },
            'recommendations': formatted_recommendations,
            'total_recommendations': len(formation_recommendations),
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        return jsonify({'error': error_msg}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model_loaded,
        'dataset_loaded': dataset is not None,
        'recommendations_available': len(all_formations) > 0,
        'dataset_size': len(dataset) if dataset is not None else 0,
        'total_formations': len(all_formations),
    })

if __name__ == "__main__":
    print("🚀 Starting Flask ML service...")
    print("📍 Service available at: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)