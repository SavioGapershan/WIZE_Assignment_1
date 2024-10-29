from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import random

nltk.download('punkt', download_dir='Home/Downloads/AI_virtusa')
nltk.download('stopwords',download_dir='Home/Downloads/AI_virtusa')

app = Flask(__name__)

# Load Model and Tokenizer
def load_model():
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# Preprocess Text
def preprocess_text(paragraph):
    sentences = sent_tokenize(paragraph)
    words = word_tokenize(paragraph.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return sentences, words

# Calculate Importance
def calculate_importance(sentences):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    importance = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1))
    return importance

# Hallucination Level Calculation
def calculate_hallucination_level(sentences, generated_text):
    hallucination_count = sum([1 for word in generated_text.split() if word not in sentences])
    hallucination_level = hallucination_count / len(generated_text.split())
    return hallucination_level

# Generate Questions
def generate_questions(model, tokenizer, sentences, importance, creativity, focus):
    questions = []
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    question_templates = [
        "What does {character} believe {subject} can reveal?",
        "According to the conversation, what are many people skeptical about regarding {subject}?",
    ]

    for word, _ in sorted_importance[:int(10 * focus)]:
        for sentence in sentences:
            if word in sentence:
                input_text = random.choice(question_templates).format(
                    character="Speaker1" if "Speaker1" in sentence else "Speaker2",
                    subject=word
                )
                input_ids = tokenizer.encode(input_text, return_tensors="pt")
                outputs = model.generate(
                    input_ids, max_length=int(50 * creativity), num_beams=5, early_stopping=True
                )
                question = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if question not in questions:
                    questions.append((question, sentence))
                if len(questions) >= 10:
                    break
        if len(questions) >= 10:
            break
    return questions

# Generate Options
def generate_options(correct_answer, sentences):
    options = [correct_answer]
    distractors = [s for s in sentences if s != correct_answer and len(s.split()) > 5]
    random.shuffle(distractors)
    options.extend(distractors[:3])
    random.shuffle(options)
    return options

# Conversation Type Classification
def classify_conversation(paragraph):
    informative_keywords = {"describe", "explain", "analyze"}
    technical_keywords = {"technology", "engineering", "science"}
    casual_keywords = {"chat", "talk", "discussion"}

    informative_score = sum(paragraph.count(word) for word in informative_keywords)
    technical_score = sum(paragraph.count(word) for word in technical_keywords)
    casual_score = sum(paragraph.count(word) for word in casual_keywords)

    if max(informative_score, technical_score, casual_score) == informative_score:
        return "Informative"
    elif max(informative_score, technical_score, casual_score) == technical_score:
        return "Technical"
    else:
        return "Casual"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        paragraph = request.form["paragraph"]
        creativity = float(request.form["creativity"])
        focus = float(request.form["focus"])
        hallucination_sensitivity = float(request.form["hallucination_sensitivity"])

        sentences, words = preprocess_text(paragraph)
        importance = calculate_importance(sentences)
        model, tokenizer = load_model()

        question_sentences = generate_questions(model, tokenizer, sentences, importance, creativity, focus)
        qa_pairs = []
        for question, sentence in question_sentences:
            correct_answer = sentence
            options = generate_options(correct_answer, sentences)
            qa_pairs.append((question, options, correct_answer))

        generated_text = " ".join([q for q, _, _ in qa_pairs])
        hallucination_level = calculate_hallucination_level(sentences, generated_text)
        adjusted_hallucination = hallucination_level * hallucination_sensitivity

        conversation_type = classify_conversation(paragraph)

        return render_template("index.html", qa_pairs=qa_pairs, hallucination_level=adjusted_hallucination, conversation_type=conversation_type)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

