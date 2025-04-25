import streamlit as st
import re
import numpy as np
import nltk
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = nltk.WordNetLemmatizer()


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatization(sentence):
    try:
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        lemmatized_words = [
            lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in pos_tags
        ]
        return ' '.join(lemmatized_words)
    except Exception as e:
        st.error(f"Error during lemmatization: {e}")
        return sentence


def preprocess(sentence):
    try:
        sentence = sentence.lower().strip()
        return lemmatization(sentence)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return sentence


@st.cache_resource
def load_model_and_tokenizer(model_name="google/pegasus-xsum",cache_dir="C:\\Users\\Vivek\\.cache\\huggingface\\"):
    try:
        tokenizer = PegasusTokenizer.from_pretrained(model_name ,cache_dir=cache_dir )
        model = PegasusForConditionalGeneration.from_pretrained(model_name ,cache_dir=cache_dir)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.stop()


def EXsummarization(text, num_sentences=5):
    try:
        sentences = nltk.sent_tokenize(text)

        processed_sent = [preprocess(sentence) for sentence in sentences]

        vectorizer = TfidfVectorizer(max_features=100000)
        matrix = vectorizer.fit_transform(processed_sent)
        matrix = matrix.toarray() if hasattr(matrix, 'toarray') else matrix

        n_features = matrix.shape[1]
        n_components = min(100, n_features)

        svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42)
        svd.fit(matrix)

        sent_score = np.dot(matrix, svd.components_.T).flatten()
        if hasattr(sent_score, 'toarray'):
            sent_score = sent_score.toarray().flatten()

        sent_score = (sent_score - np.min(sent_score)) / (np.max(sent_score) - np.min(sent_score) + 1e-6)

        ranked_sent_id = np.argsort(sent_score)[::-1]

        valid_indices = [idx for idx in ranked_sent_id if idx < len(sentences)]

        def is_similar(new_sent_vector, selected_vectors, threshold=0.7):
            if not selected_vectors:
                return False
            similarities = cosine_similarity([new_sent_vector], selected_vectors)
            return np.any(similarities[0] > threshold)

        selected_sentences = []
        selected_vectors = []
        for idx in valid_indices:
            if not is_similar(matrix[idx], selected_vectors, threshold=0.7):
                selected_sentences.append(sentences[idx])
                selected_vectors.append(matrix[idx])
            if len(selected_sentences) == num_sentences:
                break

        top_indices = [sentences.index(sentence) for sentence in selected_sentences]
        top_indices = sorted(top_indices)

        summary_sentences = [sentences[idx] for idx in top_indices]
        summary = ' '.join(summary_sentences)
        return summary
    except Exception as e:
        st.error(f"Error during extractive summarization: {e}")
        return "Error generating summary."


def determine_parameters(text_length):
    try:
        max_length = 150
        min_length = 60
        num_beams = 5
        length_penalty = 1.2
        repetition_penalty = 2.5
        num_sentences = 5

        if text_length <= 600:
            max_length = 50
            min_length = 20
            num_beams = 4
            length_penalty = 1.5
            repetition_penalty = 1.5
            num_sentences = 2
        elif text_length <= 1000:
            max_length = 80
            min_length = 30
            num_beams = 5
            length_penalty = 1.3
            repetition_penalty = 2.0
            num_sentences = 3

        elif text_length <= 1500:
            max_length = 100
            min_length = 40
            num_beams = 6
            length_penalty = 1.2
            repetition_penalty = 2.0
            num_sentences = 6

        else:
            max_length = 200
            min_length = 80
            num_beams = 7
            length_penalty = 1.0
            repetition_penalty = 2.5
            num_sentences = 10

        return max_length, min_length, num_beams, length_penalty, repetition_penalty, num_sentences
    except Exception as e:
        st.error(f"Error determining summarization parameters: {e}")
        return 150, 60, 5, 1.2, 2.5, 5, 0.7


try:
    st.title("SUMMit")
    st.write("Paste text of any length and get a concise summary")

    tokenizer, model = load_model_and_tokenizer()

    text = st.text_area("Input Text", "Paste your text here...", height=300)
    words = text.split()
    length = len(words)

    if length == 0:
        st.warning("No words detected in input. Please paste some text.")
    else:
        st.write(f"Word Count: {length}")

    if st.button("Generate Summary"):
        if text.strip() == "":
            st.warning("Please paste some text to summarize.")
        else:
            try:
                progress_bar_container = st.empty()
                progress_bar = progress_bar_container.progress(0)
                progress_step = 20
                progress_bar.progress(progress_step)

                text_length = len(text)
                max_length, min_length, num_beams, length_penalty, repetition_penalty, num_sentences = determine_parameters(
                    text_length)

                progress_step += 20
                progress_bar.progress(progress_step)

                pre_summary = EXsummarization(text, num_sentences=num_sentences)

                progress_step += 20
                progress_bar.progress(progress_step)

                pre_summary = re.sub(r'\[\d+\]', ' ', pre_summary)
                pre_summary = re.sub(r'\s+', ' ', pre_summary).strip()
                tokens = tokenizer(pre_summary, truncation=True, padding="longest", return_tensors="pt")

                progress_step += 20
                progress_bar.progress(progress_step)

                summary = model.generate(
                    **tokens,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,

                    early_stopping=True
                )
                summ = tokenizer.decode(summary[0], skip_special_tokens=True)

                progress_bar.progress(100)
                time.sleep(1)

                st.subheader("Highlights from the text:")
                st.write(pre_summary)
                st.subheader("Condensed Overview:")
                summ_cleaned = re.sub(r'\[\d+\]', ' ', summ)
                st.write(summ_cleaned)
                progress_bar_container.empty()

            except Exception as e:
                st.error(f"Error during summarization: {e}")

except Exception as main_error:
    st.error(f"An unexpected application error occurred: {main_error}")