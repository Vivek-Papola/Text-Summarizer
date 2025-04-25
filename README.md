Summary:-
SUMMit is an interactive Streamlit application that delivers both extractive and abstractive text summaries using a hybrid pipeline: it first applies TF–IDF + Truncated SVD for extractive sentence scoring, then refines with Google’s Pegasus model for an abstractive condensing step. The UI displays progress and allows parameter tuning (beam size, length penalties, number of extractive sentences), making it ideal for researchers and content creators seeking concise overviews from large texts.

Features:-
 ~ Hybrid Summarization (Extractive + Abstractive) for balanced relevance and fluency 
 ~ Dynamic Parameter Selection based on input length (auto‐adjusts max_length, num_beams, etc.) 
 ~ Lemmatized Preprocessing with NLTK for cleaner input to TF–IDF and SVD
 ~ Progress Bar & Caching of the Pegasus model for faster subsequent runs via Streamlit’s @st.cache_resource 
 ~ Error Handling in each pipeline stage, with user‐friendly messages on failures 

Tech Stack:-
 ~ Python 3.8+ – core language 
 ~ Streamlit – front-end UI framework 
 ~ NLTK – tokenization, POS tagging, and lemmatization 
 ~ scikit-learn – TF–IDF vectorization and Truncated SVD (LSA) 
 ~ Transformers – PegasusForConditionalGeneration & PegasusTokenizer from Hugging Face 
