# UiA IKT RAG Chatbot

This project implements a small Retrieval Augmented Generation (RAG) chatbot for answering questions about IKT courses at the University of Agder (UiA). The chatbot uses public UiA course pages as an external knowledge corpus and a pretrained local language model through Ollama for answer generation.

The project was developed for the IKT-469 Deep Neural Networks assignment option on RAG systems.

## Project overview

The notebook implements the full RAG pipeline:

1. Scrape public UiA IKT course pages.
2. Convert the scraped HTML content into cleaned plain text.
3. Represent each course page as a document with metadata such as course code, title and source URL.
4. Split documents into chunks using different chunking strategies.
5. Embed chunks with a sentence-transformer embedding model.
6. Store embeddings and metadata in a local Chroma vector database.
7. Retrieve relevant chunks for a user question using dense similarity search.
8. Insert retrieved context into a prompt.
9. Generate grounded answers using `llama3.1:8b` through Ollama.
10. Run experiments over different RAG settings and save the results to CSV.

## Main files

- `RAG_chatbot_with_progress_saving_gpu.ipynb`  
  Main notebook containing the scraping, RAG pipeline, experiment grid, progress saving and GPU checks.

- `rag_experiment_results.csv`  
  Final experiment results from the RAG grid search.

- `requirements.txt`  
  Python dependencies needed to run the notebook.

## Dataset

The dataset consists of 48 public UiA IKT course pages. The scraped content includes course information such as course descriptions, learning outcomes, teaching and learning methods, examination forms, credits, language, semester and prerequisites.

The scraped pages are stored locally as a JSONL file inside the data directory:

```text
uia_ikt_rag_data/uia_ikt_courses.jsonl
```

The JSONL file acts as a local snapshot of the course pages at the time of scraping. If the UiA course pages change, the corpus should be scraped again to keep the chatbot updated.

## Requirements

The project requires:

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- Ollama
- The Ollama model `llama3.1:8b`
- Python packages listed in `requirements.txt`

For GPU acceleration, a CUDA-compatible PyTorch installation is recommended. The notebook was tested with an NVIDIA GPU and CUDA-enabled PyTorch.

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama from the official Ollama website, then pull the model:

```bash
ollama pull llama3.1:8b
```

Start Ollama before running the notebook:

```bash
ollama serve
```

On Windows, Ollama may also run automatically in the background. You can check whether the model is loaded with:

```bash
ollama ps
```

Ideally, the `PROCESSOR` column should show GPU usage if GPU acceleration is configured correctly.

## Running the notebook

1. Start Ollama.
2. Open `RAG_chatbot_with_progress_saving_gpu.ipynb` in Jupyter.
3. Run the cells from top to bottom.
4. The notebook will scrape or load course data, build vector stores, run RAG experiments and save results.

The notebook saves partial progress during long experiment runs:

```text
uia_ikt_rag_data/rag_experiment_partial_results.csv
```

The final results are saved as:

```text
uia_ikt_rag_data/rag_experiment_results.csv
```

If a run is interrupted, the notebook can continue from the partial results file when resume mode is enabled.

## Experiment settings

The experiment varies five main RAG parameters:

- Chunking method: `recursive`, `markdown`
- Chunk size: `300`, `500`, `700`, `900`, `1200`
- Chunk overlap: `0`, `50`, `120`, `200`
- Retrieval depth `k`: `1`, `2`, `3`, `5`, `7`, `10`
- Prompt variant: `strict_grounded`, `friendly_advisor`, `compare_courses`

The following components are kept fixed:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: Chroma
- LLM: `llama3.1:8b` via Ollama
- Temperature: `0`
- Context length: `4096`

## Notes

- The LLM is not fine-tuned on UiA course pages. Course-specific knowledge is supplied through retrieved context from the vector database.
- The system is designed for questions about the scraped IKT course pages. It cannot reliably answer questions about information not present in the corpus, such as student satisfaction or job outcomes.
- If the project folder is inside OneDrive or another synced folder, CSV progress saving can occasionally fail due to temporary file locks. Running the project outside synced folders is recommended for long experiments.
