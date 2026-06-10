#  Cross-Domain Multi-Match Semantic Search Engine

An advanced, conceptual document retrieval system built using deep learning and Natural Language Processing (NLP). Unlike traditional keyword-based search tools, this engine understands the contextual meaning of queries to extract highly relevant text passages from multi-domain PDF matrices.

---

## Interface Preview

>  **Tip:** Replace the placeholder link below with your actual uploaded screenshot link on GitHub to display your beautiful UI!

![Semantic Search Engine UI Preview](YOUR_IMAGE_LINK_HERE)

---

##  Key Features

* **Contextual Intelligence:** Uses the state-of-the-art `all-MiniLM-L6-v2` Sentence Transformer for deep semantic vector embedding.
* **Smart Document Chunking:** Automatically scans structural sub-folders, parses raw multi-page PDF content via `pdfplumber`, and generates granular text chunks.
* **Mathematical Precision:** Computes exact relevance using Cosine Similarity scoring, ensuring the most mathematically sound matches appear first.
* **Confidence Thresholding:** Built-in structural filtering that weeds out low-relevance matches below a 20% confidence bar.
* **Dynamic Full-Width Layout:** A seamless, responsive Gradio-powered web interface optimized for deep reading and analytics.
* **Instant Document Retrieval:** Direct integration that lets you download or view the highest-ranking PDF source file natively.

---

##  System Architecture & Workflow

1.  **Ingestion:** The system dynamically structures your designated PDF directory (`/pdfs`) mapping root folders as specific intellectual "Domains."
2.  **Vectorization:** Raw text data is extracted page-by-page, cleaned, and compiled into a semantic vector corpus array.
3.  **Inference:** User queries are vectorized in real-time and mapped directly against the document tensor space.

---

##  Installation & Local Setup

### Prerequisites
Ensure you have Python 3.8+ installed on your host system.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/semantic-search-engine.git](https://github.com/your-username/semantic-search-engine.git)
cd semantic-search-engine


ඔබට කෙළින්ම පිටපත් කර (Copy) ඔබ සෑදූ `README.md` ගොනුවේ අදාළ තැනට ඇතුළත් කළ හැකි පරිදි සකස් කළ Markdown කේතය (Code) පහතින් දැක්වේ:

```markdown
### 2. Install Required Dependencies
```bash
pip install -r requirements.txt

```

### 3. Setup Your Document Directory

Create a structured directory for your PDFs to allow domain tracking:

```text
content/
└── drive/
    └── MyDrive/
        └── pdfs/
            ├── Computing/
            │   └── formal-semantics.pdf
            └── Mathematics/
                └── algebra-structures.pdf

```

### 4. Execute the Application

```bash
python app.py

```

*Once launched, open the local URL (e.g., `http://127.0.0.1:7860`) or the public Gradio live link in your web browser.*

---

## Core Technologies Used

* **Core Logic:** Python 3
* **UI Framework:** [Gradio](https://www.gradio.app/)
* **Embedding Model:** [Sentence-Transformers (Hugging Face)](https://sbert.net/)
* **PDF Extraction:** [pdfplumber](https://github.com/jsvine/pdfplumber)
* **Matrix Math:** NumPy & Scikit-Learn

```

```
