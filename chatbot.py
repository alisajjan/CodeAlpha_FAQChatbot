import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data (only first time)
nltk.download('punkt')
nltk.download('stopwords')

# FAQ dataset
faqs = {
    "What is CodeAlpha?": "CodeAlpha is an internship platform offering real-world projects.",
    "What is this chatbot for?": "This chatbot helps answer FAQ-based questions.",
    "How many tasks should I complete?": "You must complete 2 or 3 tasks for the certificate.",
    "Do I need to post on LinkedIn?": "Yes, posting your work on LinkedIn is required.",
    "How do I submit my project?": "You can submit your task through the Google Form shared on WhatsApp."
}

# Convert FAQ questions into a list
questions = list(faqs.keys())
answers = list(faqs.values())

# Function to find best match
def get_best_match(user_input):
    all_sentences = questions + [user_input]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_sentences)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    best_match_index = similarity_scores.argmax()
    return answers[best_match_index]

# Chatbot loop
print("=== CodeAlpha FAQ Chatbot ===")
print("Ask me anything or type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = get_best_match(user_input)
    print("Chatbot:", response)
