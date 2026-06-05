# 🤖 AI FAQ Chatbot

A beginner-friendly FAQ chatbot about Artificial Intelligence, built with **Python**, **NLTK**, and **Scikit-learn**. It uses **TF-IDF Vectorization** and **Cosine Similarity** to intelligently match user questions to the most relevant answers in a dataset.

---

## 📁 Project Structure

```
CodeAlpha_FAQChatbot/
│
├── chatbot.py        # Main chatbot script
├── faq.csv           # FAQ dataset (25 AI-related Q&A pairs)
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

---

## 🧠 How It Works

1. **Load Data** — Reads AI-related questions and answers from `faq.csv`.
2. **Preprocess Text** — Converts text to lowercase, removes punctuation, and tokenizes it using NLTK.
3. **TF-IDF Vectorization** — Transforms all FAQ questions into numerical vectors using Scikit-learn's `TfidfVectorizer`.
4. **Cosine Similarity** — When a user asks a question, it is vectorized and compared against all FAQ vectors. The closest match (above a similarity threshold) is returned.
5. **Fallback** — If no question is similar enough, the bot replies: *"Sorry, I don't know the answer to that question."*

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### Step 1 — Clone or Download the Project

```bash
git clone https://github.com/your-username/CodeAlpha_FAQChatbot.git
cd CodeAlpha_FAQChatbot
```

Or simply download and unzip the project folder.

### Step 2 — (Optional) Create a Virtual Environment

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the Chatbot

```bash
python chatbot.py
```

---

## 💬 Example Interaction

```
============================================================
       Welcome to the AI FAQ Chatbot!
  Ask me anything about Artificial Intelligence.
  Type 'exit' to quit.
============================================================

[INFO] Loaded 25 FAQ entries from 'faq.csv'.
[INFO] Chatbot is ready! Start asking your questions.

You: What is AI?
Bot: Artificial Intelligence (AI) is the simulation of human intelligence
     processes by computer systems. These processes include learning,
     reasoning, and self-correction...

You: Tell me about deep learning
Bot: Deep Learning is a subset of Machine Learning that uses artificial
     neural networks with many layers (deep networks) to model and
     understand complex patterns in data...

You: What is pizza?
Bot: Sorry, I don't know the answer to that question.

You: exit
Bot: Goodbye! Have a great day!
```

---

## 📊 FAQ Dataset Topics

The `faq.csv` file contains **25 Q&A pairs** covering:

| # | Topic |
|---|-------|
| 1 | What is AI? |
| 2 | Machine Learning |
| 3 | Deep Learning |
| 4 | Natural Language Processing |
| 5 | Computer Vision |
| 6 | Neural Networks |
| 7 | Supervised Learning |
| 8 | Unsupervised Learning |
| 9 | Reinforcement Learning |
| 10 | Large Language Models |
| 11 | ChatGPT |
| 12 | Transformer Models |
| 13 | Overfitting |
| 14 | Datasets |
| 15 | TF-IDF |
| 16 | Cosine Similarity |
| 17 | Chatbots |
| 18 | Algorithms |
| 19 | Data Science |
| 20 | Generative AI |
| 21 | AI Bias |
| 22 | APIs |
| 23 | Cloud Computing in AI |
| 24 | Edge AI |
| 25 | Prompt Engineering |

---

## ⚙️ Key Technologies

| Library | Purpose |
|---------|---------|
| `nltk` | Tokenization and text preprocessing |
| `scikit-learn` | TF-IDF Vectorizer and Cosine Similarity |
| `csv` (built-in) | Reading the FAQ dataset |
| `string` (built-in) | Removing punctuation |

---

## 🔧 Customization

### Add More Questions
Simply add new rows to `faq.csv` following the format:
```
question,answer
Your new question here?,Your detailed answer here.
```

### Adjust Sensitivity
In `chatbot.py`, change the `threshold` parameter in `get_best_answer()`:
```python
# Lower value = more permissive (may match loosely related questions)
# Higher value = stricter (requires closer matches)
answer = get_best_answer(user_input, vectorizer, tfidf_matrix, answers, threshold=0.3)
```

---

## 📄 License

This project is open-source and free to use for learning and educational purposes.

---

## 👤 Author

Built as part of the **CodeAlpha Python Internship**.
