# Smart RAG Assistant

This project is a prototype of a hybrid Retrieval-Augmented Generation (RAG) assistant that combines:

- Symbolic reasoning using a Knowledge Graph (RDFLib)
- Semantic search using Chroma vector store and HuggingFace embeddings
- Language-based response generation using Mistral-7B-Instruct

It supports:
- Triple extraction (regex-based)
- Pruning (importance and semantic similarity)
- Visualization of the knowledge graph (networkx and pyvis)
- Hybrid QA from both KG and vector store

## Project Files

- `smart_rag_assistant.py`: Main assistant script (CLI-based)
- `visualize_knowledge_graph.py`: Load and visualize the saved KG as interactive HTML
- `requirements.txt`: Dependencies required

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Hugging Face token (required for loading Mistral):

```bash
export HF_TOKEN=your_token_here
```

Then run the assistant:

```bash
python smart_rag_assistant.py
```

## Visualization

To visualize the KG in HTML:

```bash
python visualize_knowledge_graph.py
```

This will open/save `knowledge_graph.html`.

## Notes

- Tested with Python 3.10+
- This is a learning project; LLM-based triple extraction and FB15k-237 integration are in progress

## Environment
- Developed and tested on Google Colab with T4 GPU (15GB VRAM)
- Running Mistral-7B locally required Colab's high-RAM GPU instance
- Ensure you have access to a similar runtime if replicating the setup

