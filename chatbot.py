# =============================================================================
# AI FAQ Chatbot
# Built with Python, NLTK, and Scikit-learn
# Uses TF-IDF Vectorizer + Cosine Similarity to match user questions
# =============================================================================

import csv
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to use NLTK's word tokenizer for better text processing.
# If NLTK is not installed or data files are missing, we fall back to
# a simple whitespace split — which works perfectly fine for TF-IDF.
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import word_tokenize
    _tokenize = word_tokenize
    print("[INFO] Using NLTK tokenizer.")
except Exception:
    _tokenize = str.split
    print("[INFO] NLTK not found — using simple whitespace tokenizer (install nltk for better results).")

# =============================================================================
# STEP 1: Load the FAQ dataset from the CSV file
# =============================================================================

def load_faq(filepath="faq.csv"):
    """
    Reads the FAQ CSV file and returns two lists:
    - questions: list of all FAQ questions
    - answers:   list of corresponding answers
    """
    questions = []
    answers = []

    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)  # Read rows as dictionaries
        for row in reader:
            questions.append(row['question'])
            answers.append(row['answer'])

    print(f"[INFO] Loaded {len(questions)} FAQ entries from '{filepath}'.\n")
    return questions, answers


# =============================================================================
# STEP 2: Preprocess text
# =============================================================================

def preprocess(text):
    """
    Cleans and normalizes a text string:
    1. Converts to lowercase
    2. Removes punctuation
    3. Tokenizes into words
    4. Joins tokens back into a clean string
    """
    # Step 2a: Lowercase the entire text
    text = text.lower()

    # Step 2b: Remove all punctuation characters
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 2c: Tokenize — split the text into individual words
    tokens = _tokenize(text)

    # Step 2d: Join tokens back into a single string for TF-IDF input
    return ' '.join(tokens)


# =============================================================================
# STEP 3: Build the TF-IDF model from FAQ questions
# =============================================================================

def build_vectorizer(questions):
    """
    Creates a TF-IDF Vectorizer trained on the preprocessed FAQ questions.
    Returns:
    - vectorizer:     the fitted TfidfVectorizer object
    - tfidf_matrix:   the TF-IDF matrix of all FAQ questions
    """
    # Preprocess every question in the dataset
    processed_questions = [preprocess(q) for q in questions]

    # Initialize and fit the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_questions)

    return vectorizer, tfidf_matrix


# =============================================================================
# STEP 4: Find the best matching FAQ answer
# =============================================================================

def get_best_answer(user_input, vectorizer, tfidf_matrix, answers, threshold=0.3):
    """
    Compares the user's question against all FAQ questions using Cosine Similarity.

    Parameters:
    - user_input:    the raw question typed by the user
    - vectorizer:    the fitted TF-IDF vectorizer
    - tfidf_matrix:  TF-IDF matrix of FAQ questions
    - answers:       list of FAQ answers (parallel to questions)
    - threshold:     minimum similarity score to consider a match (default: 0.3)

    Returns:
    - The best matching answer string, or a fallback message if no match found
    """
    # Preprocess the user's input the same way we preprocessed the FAQ questions
    processed_input = preprocess(user_input)

    # Transform user input into a TF-IDF vector using the same vocabulary
    user_vector = vectorizer.transform([processed_input])

    # Compute cosine similarity between user input and all FAQ questions
    # Result shape: (1, num_questions)
    similarities = cosine_similarity(user_vector, tfidf_matrix)

    # Flatten to a 1D array and find the index of the highest similarity score
    similarity_scores = similarities.flatten()
    best_index = similarity_scores.argmax()
    best_score = similarity_scores[best_index]

    # Only return an answer if the similarity is above the threshold
    if best_score >= threshold:
        return answers[best_index]
    else:
        return "Sorry, I don't know the answer to that question."


# =============================================================================
# STEP 5: Run the chatbot in the terminal
# =============================================================================

def run_chatbot():
    """
    Main loop that runs the chatbot interactively in the terminal.
    - Loads FAQ data
    - Builds TF-IDF model
    - Accepts user input in a loop
    - Returns matched answers or a fallback message
    - Exits cleanly when user types 'exit'
    """
    print("=" * 60)
    print("       Welcome to the AI FAQ Chatbot!")
    print("  Ask me anything about Artificial Intelligence.")
    print("  Type 'exit' to quit.")
    print("=" * 60)
    print()

    # Load FAQ data from CSV
    questions, answers = load_faq("faq.csv")

    # Build the TF-IDF vectorizer and matrix
    vectorizer, tfidf_matrix = build_vectorizer(questions)
    print("[INFO] Chatbot is ready! Start asking your questions.\n")

    # Main interaction loop
    while True:
        # Get user input
        user_input = input("You: ").strip()

        # Skip empty input
        if not user_input:
            continue

        # Exit condition
        if user_input.lower() in ('exit', 'quit', 'bye'):
            print("Bot: Goodbye! Have a great day!")
            break

        # Find and display the best matching answer
        answer = get_best_answer(user_input, vectorizer, tfidf_matrix, answers)
        print(f"Bot: {answer}\n")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_chatbot()
