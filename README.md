# CodeAlpha — FAQ Chatbot  
### Intern: *Ali Sajjan*

## 📌 Project Overview  
This project is created for the **CodeAlpha Artificial Intelligence Internship — Task 2**.  
It is a simple and effective **FAQ Chatbot** that uses NLP to find the best matching answer from predefined questions.

The chatbot uses:
- TF-IDF Vectorization  
- Cosine Similarity  
- NLTK Tokenization  

It compares the user’s input with existing FAQ questions and returns the closest matching answer.

---

## 🚀 Technologies Used  
- Python  
- NLTK  
- Scikit-learn  
- VS Code  

---

## 🛠️ Features  
✔ Answers FAQ-style questions  
✔ Uses TF-IDF + Cosine Similarity for matching  
✔ Automatically finds the closest question  
✔ Beginner-friendly AI/NLP project  

---

## 📎 Installation

Install required libraries:

pip install nltk scikit-learn

yaml
Copy code

---

## 🧠 Code Snippet

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

faqs = {
"What is CodeAlpha?": "CodeAlpha is an internship platform offering real-world projects.",
"What is this chatbot for?": "This chatbot helps answer FAQ-based questions.",
"How many tasks should I complete?": "You must complete 2 or 3 tasks for the certificate.",
"Do I need to post on LinkedIn?": "Yes, posting your work on LinkedIn is required.",
"How do I submit my project?": "You can submit your task through the Google Form shared on WhatsApp."
}

questions = list(faqs.keys())
answers = list(faqs.values())

def get_best_match(user_input):
all_sentences = questions + [user_input]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_sentences)
similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
best_match_index = similarity_scores.argmax()
return answers[best_match_index]

print("=== CodeAlpha FAQ Chatbot ===")
print("Ask me anything or type 'exit' to quit.\n")

while True:
user_input = input("You: ")
if user_input.lower() == "exit":
print("Chatbot: Goodbye!")
break
response = get_best_match(user_input)
print("Chatbot:", response)

yaml
Copy code

---

## 🎯 Example Interaction

You: What is CodeAlpha?
Bot: CodeAlpha is an internship platform offering real-world projects.

You: How many tasks should I complete?
Bot: You must complete 2 or 3 tasks for the certificate.

yaml
Copy code

---

## 🎥 Demo Video  
(Will be added after recording)

---

## 🏅 Internship Task  
This project fulfills **Task 2 — Chatbot for FAQs** for CodeAlpha AI Internship.

---

## 📬 Contact  
GitHub: https://github.com/alisajjan  

---

## ✔️ Submission Status  
✅ Source code uploaded  
⬜ Demo video pending  
⬜ LinkedIn post pending 
