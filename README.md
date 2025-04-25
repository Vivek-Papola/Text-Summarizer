
# SUMMit - text summarizing tool

SUMMit is an interactive Streamlit application that delivers both extractive and abstractive text summaries using a hybrid pipeline: it first applies TF–IDF + Truncated SVD for extractive sentence scoring, then refines with Google’s Pegasus model for an abstractive condensing step. The UI displays progress and allows parameter tuning (beam size, length penalties, number of extractive sentences), making it ideal for researchers and content creators seeking concise overviews from large texts.

## Features

- Hybrid Summarization (Extractive + Abstractive) for balanced relevance and fluency
- Dynamic Parameter Selection based on input length (auto‐adjusts max_length, num_beams, etc.)
- Lemmatized Preprocessing with NLTK for cleaner input to TF–IDF and SVD
- Progress Bar & Caching of the Pegasus model for faster subsequent runs via Streamlit’s @st.cache_resource
- Error Handling in each pipeline stage, with user‐friendly messages on failures


## Tech Stack

**Python 3.8+:** core language

**Streamlit:** front-end UI framework

**NLTK:** tokenization, POS tagging, and lemmatization

**scikit-learn:** TF–IDF vectorization and Truncated SVD (LSA)

**Transformers:** PegasusForConditionalGeneration & PegasusTokenizer from Hugging Face

## Installation

Clone the repo

```bash
  git clone https://github.com/your-username/SUMMit.git  
  cd SUMMit  
```
Install dependencies
```bash
  pip install -r requirements.txt  
```
Run the app
```bash
  streamlit run app.py    
```
The app will open in your default browser at localhost:8501
## Usage/Examples

   1. Paste or upload text into the input area.

   2. Click “Generate Summary” to see:

- Highlights: top extractive sentences

- Condensed Overview: final abstractive summary

3. Adjust parameters (in sidebar) for longer/shorter summaries or different beam search settings


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository

2. Create a feature branch(`git checkout -b feature/MyFeature`)

3. Commit your changes(`git commit -m "Add MyFeature"`)

4. Push to the branch(`git push origin feature/MyFeature`)

5. Open a Pull Request

Please adhere to this project's `code of conduct`.


## License

[MIT](https://choosealicense.com/licenses/mit/) - 
This project is licensed under the MIT License – see the LICENSE file for details.

## Authors

- [@vivek-papola](https://www.github.com/vivek-papola)
Feel free to star ⭐ the repo if you find SUMMit useful!
