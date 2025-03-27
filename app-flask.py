import os
import sqlite3
import pandas as pd
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
DB_USERNAME = os.getenv("USERNAME")
DB_PASSWORD = os.getenv("PASSWORD")
LLM_API_ENDPOINT = os.getenv("LLM_API")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# Initialize Flask app
app = Flask(__name__)

class IncidentRAG:
    def __init__(self, db_path="it_incidents.db"):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.connect_to_db()
        self.load_incident_data()

    def connect_to_db(self):
        try:
            self.conn = sqlite3.connect(
                f"sqlite:///{DB_USERNAME}:{DB_PASSWORD}@localhost/{self.db_path}"
            )
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def load_incident_data(self):
        query = """
            SELECT incident_id, description, resolution, status, created_date
            FROM IT_INCIDENTS
            WHERE status = 'resolved'
        """
        self.incidents_df = pd.read_sql_query(query, self.conn)
        self.incident_embeddings = self.embedding_model.encode(
            self.incidents_df['description'].tolist(),
            batch_size=32,
            show_progress_bar=True
        )

    def call_llm_api(self, prompt):
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.7
        }
        try:
            response = requests.post(LLM_API_ENDPOINT, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['text']
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return "Error processing request"

    def find_similar_incidents(self, new_incident_desc, top_k=3):
        new_embedding = self.embedding_model.encode([new_incident_desc])[0]
        similarities = cosine_similarity(
            [new_embedding],
            self.incident_embeddings
        )[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        similar_incidents = self.incidents_df.iloc[top_indices]
        similarity_scores = similarities[top_indices]
        return similar_incidents, similarity_scores

    def generate_response(self, question, new_incident_desc=None):
        if "list" in question.lower() or "how many" in question.lower():
            return self.handle_basic_query(question)
        elif new_incident_desc:
            return self.suggest_resolution(question, new_incident_desc)
        else:
            return self.handle_basic_query(question)

    def handle_basic_query(self, question):
        prompt = f"""
        Given the IT_INCIDENTS table with columns:
        incident_id, description, resolution, status, created_date
        
        Answer: {question}
        """
        return self.call_llm_api(prompt)

    def suggest_resolution(self, question, new_incident_desc):
        similar_incidents, scores = self.find_similar_incidents(new_incident_desc)
        context = "Similar past incidents:\n"
        for idx, (index, row) in enumerate(similar_incidents.iterrows()):
            context += f"""
            Incident {idx + 1} (Similarity: {scores[idx]:.2f}):
            Description: {row['description']}
            Resolution: {row['resolution']}
            """
        
        prompt = f"""
        New incident: {new_incident_desc}
        Question: {question}
        
        {context}
        
        Suggest a resolution for the new incident.
        """
        return self.call_llm_api(prompt)

    def close(self):
        self.conn.close()

# Initialize RAG system
rag = IncidentRAG()

# Flask routes
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    question = data.get('question', '')
    new_incident = data.get('new_incident', '')
    
    try:
        response = rag.generate_response(question, new_incident)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# HTML template (save as templates/index.html)
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>IT Incident Analyzer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: auto; }
        textarea { width: 100%; margin: 10px 0; }
        button { padding: 10px 20px; }
        #response { margin-top: 20px; padding: 10px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="container">
        <h1>IT Incident Analyzer</h1>
        <form id="queryForm">
            <textarea id="question" rows="3" placeholder="Enter your question (e.g., 'How many incidents this month?' or 'How to resolve this?')"></textarea>
            <textarea id="new_incident" rows="3" placeholder="Enter new incident description (optional)"></textarea>
            <button type="submit">Submit</button>
        </form>
        <div id="response"></div>
    </div>

    <script>
        document.getElementById('queryForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const question = document.getElementById('question').value;
            const new_incident = document.getElementById('new_incident').value;
            
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, new_incident })
            });
            
            const data = await response.json();
            document.getElementById('response').innerHTML = data.response || data.error;
        });
    </script>
</body>
</html>
"""

# Create templates directory and save HTML
if not os.path.exists('templates'):
    os.makedirs('templates')
with open('templates/index.html', 'w') as f:
    f.write(html_template)

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        rag.close()
