from flask import Flask, request, jsonify, render_template
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pytesseract
import cv2
import os
from docx import Document
import PyPDF2
from io import BytesIO

# Initialize Flask application
app = Flask(__name__)

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Download NLTK corpus (first time only)
nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to preprocess text
def preprocess_text(text):
    return text.lower()  # You can add more preprocessing steps if needed

# Function to perform sentiment analysis on each paragraph
def analyze_paragraphs(text):
    # Tokenize text into sentences
    sentences = nltk.sent_tokenize(text)
    # Merge smaller sentences with the next longer sentence
    merged_sentences = []
    i = 0
    while i < len(sentences):
        current_sentence = sentences[i]
        if len(current_sentence) < 30:  # Adjust the threshold for merging as needed
            if i + 1 < len(sentences):
                next_sentence = sentences[i + 1]
                merged_sentences.append(current_sentence + ' ' + next_sentence)
                i += 1  # Skip the next sentence as it has been merged
            else:
                merged_sentences.append(current_sentence)
        else:
            merged_sentences.append(current_sentence)
        i += 1
    # Perform sentiment analysis on each merged sentence
    sentiments = []
    for sentence in merged_sentences:
        # Preprocess sentence
        processed_sentence = preprocess_text(sentence)
        # Perform sentiment analysis
        scores = analyzer.polarity_scores(processed_sentence)
        # Classify sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'Favorable'
        elif scores['compound'] <= -0.05:
            sentiment = 'Unfavorable'
        else:
            sentiment = 'Neutral'
        # Store sentiment for the sentence
        sentiments.append({"sentence": sentence, "sentiment": sentiment})
    return sentiments

# Function to preprocess image before text extraction
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image file not found")
    # Preprocess the image (for example, resize and convert to grayscale)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# Function to extract text from image using Tesseract OCR
def extract_text_from_image(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    # Perform text extraction
    text = pytesseract.image_to_string(processed_image)
    return text

# Function to read text from a file
def read_text_from_file(filename):
    try:
        with open(filename, 'r') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        return "File not found"
    
@app.route('/')
def hello():
    return render_template("base.html")


# Route for text sentiment analysis
@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    # Get the text data from the request
    text_data = request.files.get('text')
    if text_data:
        # Get the filename
        filename = text_data.filename
        # Read text from the file
        text_data2 = read_text_from_file(filename)
        paragraph_sentiments = analyze_paragraphs(text_data2)
        return render_template("base.html", results=paragraph_sentiments)
    else:
        return "<h1>No text file provided</h1>"

# Route for image sentiment analysis
@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    # Get the image file from the request
    image_file = request.files.get('image')
    if image_file is None:
        return "<h1>No image file provided</h1>"
    else:
        # Save the image to a temporary file
        image_path = 'temp_image.png'
        image_file.save(image_path)
        try:
            # Extract text from the image
            text = extract_text_from_image(image_path)
            # Analyze the text sentiment
            paragraph_sentiments = analyze_paragraphs(text)
            return render_template("base.html", results=paragraph_sentiments)
        except FileNotFoundError as e:
            return str(e)

if __name__ == '__main__':
    app.run(debug=True)
