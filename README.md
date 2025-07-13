# ⚖️ Legal Insight AI

_Unlock Legal Insights with Intelligent Precision_

[![Last Commit](https://img.shields.io/github/last-commit/barnie-moses/Legal-InsightAI)](https://github.com/barnie-moses/Legal-InsightAI)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📖 Table of Contents

- [✨ Overview](#-overview)
- [🎬 Demo](#-overview)
- [🔧 Features](#-features)
- [📂 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
  - [✅ Prerequisites](#-prerequisites)
  - [📦 Installation](#-installation)
  - [💻 Usage](#-usage)
- [🧪 Testing](#-testing)
- [⚠️ Disclaimer](#%EF%B8%8F-disclaimer)
- [🤝 References](#-references)
---

## ✨ Overview

**Legal Insight AI** is an advanced AI-powered toolkit designed for intelligent legal document analysis. It transforms legal research by enabling natural language querying of complex legal documents, specifically tailored for the *Basic Laws and Authorities of the National Archives and Records Administration (NARA) 2016 edition*.

The system combines cutting-edge technologies:
- Vector embeddings for semantic understanding
- Advanced document chunking for context preservation
- Large Language Models (LLMs) for precise answer generation

---
## 🎬 Demo
[Click to watch a Demo](./assets/demo.mp4)


## 🔧 Features

### 🧠 Intelligent Legal Analysis
- Semantic search powered by FAISS vector store
- Context-aware question answering
- Precision-focused legal document parsing

### 🛠️ Technical Capabilities
- Built on LangChain for modular AI pipelines
- Utilizes HuggingFace's legal embeddings
- Efficient PDF processing with PyPDFLoader
- Local LLM execution via Ollama

### 📈 Scalable Architecture
- Designed for expansion to other legal domains
- Modular components for easy customization
- Optimized for both development and production use

---

## 📂 Project Structure

```text
Legal-InsightAI/
│
├── data/                         # Document storage
│   └── basic_laws_2016.pdf       # Primary legal document
│
├── database/                     # Vector database storage
│   └── faiss_basic_laws/         
│       ├── index.faiss          # FAISS index file
│       └── index.pkl            # Index metadata
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_legal_Insight_AI.ipynb   # Exploration notebook
│   └── 02_legal_Insight_AI.ipynb   # Development notebook
│
├── scripts/                      
│   └── app.py                    # Streamlit application
│
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
├── README.md                    # This document
└── requirements.txt             # Python dependencies
```

## 🚀 Getting Started

### ✅ Prerequisites

Before installation, ensure your system meets these requirements:

- **Python 3.10 or higher** ([Download Python](https://www.python.org/downloads/))
- **Ollama** installed for local LLM execution ([Installation Guide](https://ollama.com/))
- **Git** for version control ([Git Downloads](https://git-scm.com/downloads))
- **RAM** for optimal performance
- **Free disk space** for models and vector stores

---

### 📦 Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/barnie-moses/Legal-InsightAI
cd Legal-InsightAI
```

## 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Linux/macOS:
source venv/bin/activate

# Windows (Command Prompt):
venv\Scripts\activate.bat

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Alternatively using Conda:
conda create -n legal-ai python=3.10
conda activate legal-ai
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```
## 4. Download Models (Optional)
```bash
ollama pull llama3  # Or your preferred LLM
```

## 💻 Usage

### Launch the Application
```bash
streamlit run scripts/app.py
```

## Application Workflow

### Document Processing
- **Downloads** the legal PDF (if missing)
- **Parses and chunks** the document using `PyPDFLoader`
- **Generates embeddings** with HuggingFace models

### Vector Database
- **Creates or loads** FAISS vector store
- **Indexes** document chunks for semantic search

### Interactive Interface
- **Accepts** natural language legal queries
- **Provides** context-aware responses via LLM

## Sample Queries to Try
```text
"What are the Archivist's responsibilities regarding custody of records?"
"How are Vice-Presidential and Presidential records handled?"
"Summarize NARA's primary functions"
"What provisions exist for record preservation?"
"Explain Section 5 of the NARA laws"
```

## Interactive Testing

1. **Launch the Streamlit app**
2. **Test with various query types**:
   ```text
   "Define archival responsibilities"
   "List all legal authorities mentioned"
   ```

## Notebook Examples
- **Review development patterns** in:  
  `/notebooks/01_legal_Insight_AI.ipynb`
  `/notebooks/02_legal_Insight_AI.ipynb`
- **Experiment** with different embedding models in the notebooks

---

## ⚠️ Disclaimer

### Legal Notice
❗ **Not Legal Advice**  
This tool provides informational support only. For official matters:
- 👨⚖️ Consult qualified attorneys
- 🔍 Verify against primary sources
- ⚠️ Never rely solely on AI-generated content

### Data Provenance
- **Source**: Basic Laws and Authorities of the National Archives (2016 Edition)
- **Availability**: Publicly available through [NARA](https://www.archives.gov/)
- **Rights**: Developer claims no rights over referenced materials

### Limitations
| Limitation | Description |
|-----------|------------|
| 🎯 Accuracy | Depends on source document quality |
| 🔄 Timeliness | May not reflect recent legal changes |
| 🔍 Verification | LLM outputs require human review |

### Production Recommendations
```text
1. 🔄 Regular document updates
2. ✔️ Output validation mechanisms  
3. ⚖️ Professional legal review
```

## 📚 References

### Legal Source
- **Primary Document**:  
  _Basic Laws and Authorities of the National Archives and Records Administration (2016 Edition)_  
  Acquired via public access channels from [NARA](https://www.archives.gov/)

### Technology Stack
| Component | Purpose |
|-----------|---------|
| [LangChain](https://python.langchain.com) | AI orchestration framework |
| [Ollama](https://ollama.ai) | Local LLM execution |
| [HuggingFace](https://huggingface.co) | Embedding models |
| [Streamlit](https://streamlit.io) | Web interface |

### Disclaimer Notice
```text
This assistant provides informational support only and does not constitute legal advice. 
Content is derived from publicly available documents. The developer claims no legal 
authority or ownership over referenced materials.
```

## 🚀 Powered By

<div align="center">
  <a href="https://python.langchain.com"><img src="https://img.shields.io/badge/LangChain-FF6B4D?style=for-the-badge&logo=langchain&logoColor=white" height="30"></a>
  <a href="https://github.com/facebookresearch/faiss"><img src="https://img.shields.io/badge/FAISS-00B4D8?style=for-the-badge&logo=facebook&logoColor=white" height="30"></a>
  <a href="https://ollama.ai"><img src="https://img.shields.io/badge/Ollama-7C3AED?style=for-the-badge&logo=ollama&logoColor=white" height="30"></a>
  <a href="https://huggingface.co"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" height="30"></a>
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" height="30"></a>
</div>