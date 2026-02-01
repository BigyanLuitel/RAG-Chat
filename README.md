# 🎓 Orchid International College - AI Assistant

A conversational AI chatbot powered by **RAG (Retrieval-Augmented Generation)** techniques to provide accurate information about Orchid International College.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge)

## ✨ Features

- 🤖 **AI-Powered Responses** - Uses Llama 3.2 via Ollama for natural conversations
- 📚 **RAG Architecture** - Retrieves relevant context from college documents
- 🔍 **Semantic Search** - ChromaDB vector store with HuggingFace embeddings
- 🛡️ **Security** - Built-in prompt injection protection
- 💬 **Chat History** - Maintains conversation context
- 🎨 **Modern UI** - Beautiful Streamlit interface

## 📁 Project Structure

```
RAG-Chat/
├── app.py                 # Streamlit UI application
├── core/
│   ├── answer.py          # RAG question-answering logic
│   ├── ingest.py          # Document ingestion pipeline
│   ├── evaluation.py      # MRR evaluation module
│   └── security.py        # Prompt injection detection
├── OIC_Website/           # Knowledge base (Markdown files)
│   ├── 01_About_Us.md
│   ├── 02_BSc_CSIT.md
│   ├── 03_BCA.md
│   ├── 04_BITM.md
│   ├── 05_BBM.md
│   ├── 06_BBS.md
│   ├── 07_BSW.md
│   └── 08_Contact.md
├── vector_db/             # ChromaDB vector store (generated)
├── evaluation_data.json   # Test queries for MRR evaluation
├── .env                   # Environment variables
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/RAG-Chat.git
   cd RAG-Chat
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull the Ollama model**
   ```bash
   ollama pull llama3.2
   ```

5. **Ingest documents** (create vector embeddings)
   ```bash
   python core/ingest.py
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

7. Open your browser at `http://localhost:8501`

## ⚙️ Configuration

Create a `.env` file in the root directory:

```env
OLLAMA_MODEL=llama3.2
```

### Available Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2` | Ollama model to use for generation |

## 📖 Adding Knowledge

1. Add your Markdown files to the `OIC_Website/` directory
2. Run the ingestion script:
   ```bash
   python core/ingest.py
   ```
3. Restart the Streamlit app

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **LLM** | Ollama (Llama 3.2) |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |
| **Vector Store** | ChromaDB |
| **Framework** | LangChain |

## 📝 Usage

Simply type your question in the chat input:

- "What programs does Orchid International College offer?"
- "Tell me about BSc CSIT program"
- "What are the admission requirements?"
- "How can I contact the college?"

The assistant will retrieve relevant information from the knowledge base and provide accurate answers.

## 🔒 Security

The chatbot includes built-in protection against:
- Prompt injection attacks
- Jailbreak attempts
- Role manipulation

## � RAG Evaluation (MRR Accuracy)

The project includes a comprehensive evaluation module to measure retrieval quality using **Mean Reciprocal Rank (MRR)**.

### What is MRR?

MRR (Mean Reciprocal Rank) measures how well the retrieval system ranks relevant documents:
- **MRR = 1.0**: Relevant document is always first
- **MRR = 0.5**: Relevant document is at position 2 on average
- **MRR = 0.33**: Relevant document is at position 3 on average

### Running Evaluation

```bash
# Run the full MRR evaluation
python core/evaluation.py
```

### Output Example

```
============================================================
            RAG EVALUATION REPORT - MRR ACCURACY
============================================================

📊 SUMMARY METRICS
────────────────────────────────────────
  Mean Reciprocal Rank (MRR): 0.8500
  Hit Rate:                   93.33%
  Total Queries:              15
  Hits (relevant found):      14
  Misses (not found):         1
```

### Customizing Test Data

Edit `evaluation_data.json` to add your own test queries:

```json
[
  {
    "query": "Your test question here",
    "relevant_sources": ["expected_file.md"]
  }
]
```

### Using Evaluation in Code

```python
from core.evaluation import evaluate_mrr, evaluate_mrr_at_k

# Run basic evaluation
report = evaluate_mrr(k=10)
print(f"MRR Score: {report.mrr_score}")

# Evaluate at different k values
mrr_scores = evaluate_mrr_at_k(k_values=[1, 3, 5, 10])
```

## �📄 License

This project is licensed under the MIT License.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Made with ❤️ for Orchid International College