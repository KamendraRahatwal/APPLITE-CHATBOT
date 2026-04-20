from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import pickle
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# =========================================
# LOAD EVERYTHING ON STARTUP
# =========================================
print("Loading index...")
index = faiss.read_index("faq.index")

print("Loading records...")
with open("records.pkl", "rb") as f:
    records = pickle.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Embedding device:", device)

embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print(f"✅ API Ready! Loaded {len(records)} records")

# =========================================
# GREETING & SMALL TALK FILTERS
# =========================================
GREETINGS = {
    "hi", "hello", "hey", "hii", "hiii",
    "good morning", "good afternoon", "good evening", "good night",
    "howdy", "what's up", "whats up", "sup"
}

THANKS = {
    "thanks", "thank you", "thankyou", "thank u",
    "thx", "ty", "much appreciated", "appreciate it"
}

GOODBYES = {
    "bye", "goodbye", "see you", "see ya",
    "take care", "later", "cya", "no thanks",
    "no thank you", "i'm good", "im good", "that's all",
    "thats all", "nothing else", "nope", "nah"
}

def is_greeting(q): return q.lower().strip() in GREETINGS
def is_thanks(q):   return q.lower().strip() in THANKS
def is_goodbye(q):  return q.lower().strip() in GOODBYES

# =========================================
# FAISS SEARCH
# =========================================
SIM_THRESHOLD = 0.25

def search_context(query, k=5):
    emb = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(emb)
    scores, indices = index.search(emb, k)
    if scores[0][0] < SIM_THRESHOLD:
        return None
    contexts = []
    for idx in indices[0]:
        r = records[idx]
        contexts.append(f"\nIntent: {r['intent']}\nKeywords: {r['keywords']}\nAnswer: {r['answer']}\n")
    return "\n".join(contexts)

# =========================================
# CHAT ENDPOINT
# =========================================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    q = data.get("message", "").strip()

    if not q:
        return jsonify({"reply": "Please type a message."})

    # Handle greetings
    if is_greeting(q):
        return jsonify({"reply": "Hello! Welcome to Applite Solutions. How can I assist you today?"})

    # Handle thanks
    if is_thanks(q):
        return jsonify({"reply": "You're welcome! Is there anything else I can help you with?"})

    # Handle goodbyes
    if is_goodbye(q):
        return jsonify({"reply": "Goodbye! Have a great day. Feel free to reach out anytime you need help!"})

    # Handle simple conversational responses
    q_lower = q.lower().strip()
    
    if q_lower in ["no", "nope", "nah", "nothing", "not really", "im good", "i'm good"]:
        return jsonify({"reply": "Alright! Feel free to reach out if you need anything. Have a great day!"})
    
    if q_lower in ["yes", "yeah", "yep", "sure", "okay", "ok", "yup"]:
        return jsonify({"reply": "Great! What would you like to know about? You can ask me about medical billing, credentialing, coding, or any of our services."})

    # Search FAQ
    context = search_context(q)

    if context is None:
        return jsonify({"reply": "I'm sorry, I couldn't find that information. Please contact Applite Solutions support for further assistance."})

    # Get the matched records for follow-up questions
    emb = embed_model.encode([q]).astype("float32")
    faiss.normalize_L2(emb)
    scores, indices = index.search(emb, 5)
    
    matched_records = []
    for idx in indices[0]:
        matched_records.append(records[idx])

    # Build prompt and call OpenAI
    prompt = f"""You are Applite Solutions official support assistant.

Rules:
- Answer based on the information provided below
- Be helpful and professional
- If information contains contact details, phone numbers, or emails, include them in your response
- Keep responses clear and under 3-4 sentences
- Do NOT say you don't have information if context is provided

Context from knowledge base:
{context}

User Question: {q}

Answer:"""

    output = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.2
    )

    answer = output.choices[0].message.content.strip()
    
    # Extract follow-up questions from the best matched record
    follow_ups = []
    if matched_records:
        first_record = matched_records[0]
        
        # Check for follow_up_questions key (with underscore)
        follow_up_text = first_record.get('follow_up_questions', '')
        
        if follow_up_text and follow_up_text != 'N/A' and follow_up_text.strip() != '':
            # Split by | symbol
            if '|' in follow_up_text:
                follow_ups = [q.strip() for q in follow_up_text.split('|') if q.strip()]
            # Or split by comma if no pipe
            elif ',' in follow_up_text:
                follow_ups = [q.strip() for q in follow_up_text.split(',') if q.strip()]
            else:
                # Single follow-up question
                follow_ups = [follow_up_text.strip()]
            
            # Limit to 3 follow-ups
            follow_ups = follow_ups[:3]
    
    # Check for form trigger phrases
    contact_phrases = [
        "i want", "i need", "get a demo", "request a demo", "need a quote",
        "get pricing", "contact me", "call me", "reach out", "speak to",
        "talk to someone", "get in touch", "schedule", "book a call"
    ]
    
    should_show_form = any(phrase in q_lower for phrase in contact_phrases)
    
    if should_show_form or "contact" in answer.lower():
        matched_intent = "general"
        if matched_records:
            matched_intent = matched_records[0]['intent']
        
        return jsonify({
            "reply": "We're happy to help with that! To provide you with the best assistance, please share your details so our specialist can contact you.",
            "show_form": True,
            "intent": matched_intent
        })
    
    # Build response
    response = {"reply": answer}
    if follow_ups:
        response["follow_ups"] = follow_ups
    
    return jsonify(response)


# =========================================
# LEAD SUBMISSION ENDPOINT
# =========================================
@app.route("/submit-lead", methods=["POST"])
def submit_lead():
    data = request.get_json()
    
    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()
    company = data.get("company", "").strip()
    message = data.get("message", "").strip()
    intent = data.get("intent", "general").strip()
    
    if not name or not email or not phone:
        return jsonify({"success": False, "error": "Name, email, and phone are required"}), 400
    
    lead_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Intent": intent,
        "Name": name,
        "Email": email,
        "Phone": phone,
        "Company": company if company else "N/A",
        "Message": message if message else "N/A"
    }
    
    excel_file = "leads.xlsx"
    
    try:
        if os.path.exists(excel_file):
            df_existing = pd.read_excel(excel_file)
            df_new = pd.DataFrame([lead_data])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(excel_file, index=False)
        else:
            df = pd.DataFrame([lead_data])
            df.to_excel(excel_file, index=False)
        
        print(f"✅ Lead saved: {name} ({intent})")
        
        return jsonify({
            "success": True, 
            "message": "Thank you! Our team will contact you shortly."
        })
        
    except Exception as e:
        print(f"❌ Error saving lead: {e}")
        return jsonify({"success": False, "error": "Failed to save your information"}), 500


# =========================================
# HEALTH CHECK
# =========================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Applite chatbot API is running!"})


# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



