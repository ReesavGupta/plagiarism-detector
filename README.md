# Plagiarism Detector Web Application

A web-based tool for detecting potential plagiarism or text cloning using semantic similarity analysis with multiple embedding models.

## Features

- **Dynamic Text Input:** Add or remove multiple text input boxes for flexible comparison.
- **Semantic Similarity:** Uses state-of-the-art embedding models (e.g., Sentence-Transformers, OpenAI) to compute semantic similarity between texts.
- **Similarity Matrix:** Visualizes pairwise similarity percentages between all input texts.
- **Clone Detection:** Highlights text pairs with high similarity (e.g., >80%) as potential clones.
- **Model Comparison:** Compare results across different embedding models to evaluate their effectiveness.

## Core Components

1. **Text Preprocessing:**
   - Cleans and normalizes input texts for consistent embedding generation.
2. **Embedding Generation:**
   - Supports multiple models (e.g., Sentence-Transformers, OpenAI embeddings).
   - Easily extendable to add more models.
3. **Pairwise Similarity Calculation:**
   - Computes cosine similarity between all pairs of text embeddings.
   - Displays results as a similarity matrix (percentages).
4. **Results Visualization:**
   - Interactive matrix with color-coded similarity scores.
   - Highlights pairs above a configurable threshold (default: 80%).
5. **Clone Detection:**
   - Flags and lists text pairs with high similarity as potential clones.
6. **Model Comparison Report:**
   - Side-by-side comparison of similarity results from different embedding models.

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd plagiarism-detector
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or if using pyproject.toml
   pip install .
   ```

### Running the Application
```bash
streamlit main.py
```
The web interface will be available at `http://localhost:5000` (or as specified in the output).

## Usage
1. Enter or paste multiple texts into the input boxes.
2. Select the embedding model(s) to use for comparison.
3. Click "Compare" to generate the similarity matrix.
4. Review the matrix and highlighted pairs for potential clones.
5. View the model comparison report for insights on embedding performance.

## Documentation
- **How Embeddings Detect Plagiarism:**
  - Embedding models convert texts into high-dimensional vectors capturing semantic meaning.
  - Cosine similarity measures how close two vectors (texts) are in meaning.
  - High similarity scores indicate potential plagiarism or text cloning, even if wording differs.
- **Extending Models:**
  - Add new embedding models by implementing a wrapper in the embeddings module.
  - Update the UI to include new model options.

## Project Structure
- `main.py` — Application entry point
- `requirements.txt` / `pyproject.toml` — Dependencies
- `README.md` — Project documentation (this file)
- `templates/` — Web UI templates (if using Flask/FastAPI)
- `static/` — CSS/JS assets
- `embeddings/` — Embedding model wrappers
- `utils/` — Utility functions (preprocessing, similarity, etc.)

## License
MIT License

## Acknowledgements
- [Sentence-Transformers](https://www.sbert.net/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Flask](https://flask.palletsprojects.com/) or [FastAPI](https://fastapi.tiangolo.com/) for web interface
