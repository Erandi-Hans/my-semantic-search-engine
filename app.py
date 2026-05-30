# ==============================================================================
# 1. INSTALL REQUIRED LIBRARIES
# ==============================================================================
# - sentence-transformers: For semantic vector embeddings
# - pdfplumber: For highly accurate PDF text extraction
# - gradio: For building the web application interface
# ==============================================================================
!pip install -q sentence-transformers pdfplumber gradio

import os
import pdfplumber
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 2. INITIALIZE THE SEMANTIC EMBEDDING MODEL
# ==============================================================================
# We use 'all-MiniLM-L6-v2', the industry standard for fast and lightweight
# semantic search on standard CPU environments.
# ==============================================================================
print("Loading Embedding Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

PDF_FOLDER = "/content/pdfs"
all_chunks = []
corpus_embeddings = None

# Create the folder automatically if it doesn't exist
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)
    print(f"Created '{PDF_FOLDER}' folder. Please upload your domain sub-folders and PDFs into it.")

# ==============================================================================
# 3. PDF PROCESSING & SEMANTIC CHUNKING PIPELINE
# ==============================================================================
def process_pdfs():
    global all_chunks, corpus_embeddings
    all_chunks = []

    print("Scanning folders and processing PDFs...")

    # Traverse through all domain sub-folders inside /content/pdfs
    for root, dirs, files in os.walk(PDF_FOLDER):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                domain_name = os.path.basename(root)  # The sub-folder name represents the domain

                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        for page_num, page in enumerate(pdf.pages):
                            text = page.extract_text()
                            if text:
                                # Break page content into logical paragraphs (Semantic Chunking)
                                paragraphs = text.split("\n\n")
                                for para in paragraphs:
                                    # Filter out noise or extremely short sentences
                                    if len(para.strip()) > 30:
                                        all_chunks.append({
                                            "pdf_path": pdf_path,
                                            "pdf_name": file,
                                            "domain": domain_name,
                                            "page": page_num + 1,
                                            "text": para.strip()
                                        })
                except Exception as e:
                    print(f"Error reading file {file}: {e}")

    if all_chunks:
        print(f"Extracted {len(all_chunks)} semantic text chunks.")
        corpus_texts = [chunk["text"] for chunk in all_chunks]

        print("Generating Semantic Vector Embeddings (Converting text to numbers)...")
        corpus_embeddings = model.encode(corpus_texts, show_progress_bar=True)
        print("Pipeline initialization complete! System is ready for search.")
    else:
        print("System warning: No PDF files detected inside /content/pdfs yet.")

# Run the ingestion framework
process_pdfs()

# ==============================================================================
# 4. CORE SEMANTIC SEARCH LOGIC (COSINE SIMILARITY)
# ==============================================================================
def search_and_display(query):
    if not all_chunks:
        return "<h3 style='color:red;'>Error: Please upload PDFs to the 'pdfs' folder first.</h3>", None, "", ""
    if not query.strip():
        return "<h3>Please enter a search query.</h3>", None, "", ""

    # Step 1: Convert user query into a vector embedding
    query_embedding = model.encode([query])

    # Step 2: Compute mathematical similarity scores between query and document chunks
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    # Step 3: Extract the index of the highest scoring chunk
    best_index = similarities.argsort()[-1]
    score = similarities[best_index]
    best_match = all_chunks[best_index]

    # Step 4: Apply a confidence threshold to filter irrelevant inputs
    if score < 0.20:
        return """
        <div style='background-color: #ffebd2; padding: 15px; border-radius: 8px; border-left: 5px solid #ff9800;'>
            <h3 style='margin: 0; color: #cc7a00;'>Low Confidence Match</h3>
            <p>No highly relevant semantic content was discovered for this query in the document matrix.</p>
        </div>
        """, None, "", ""

    # Step 5: Format metadata as a clean HTML Dashboard card
    details_html = f"""
    <div style='background-color: #e8f4fd; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3; font-family: sans-serif;'>
        <h3 style='margin-top: 0; color: #0d47a1;'>Best Semantic Match Found</h3>
        <table style='width: 100%; border-collapse: collapse;'>
            <tr><td style='padding: 4px 0;'><b>Folder Domain:</b></td><td style='color: #1565c0;'>{best_match['domain']}</td></tr>
            <tr><td style='padding: 4px 0;'><b>Document Name:</b></td><td>{best_match['pdf_name']}</td></tr>
            <tr><td style='padding: 4px 0;'><b>Detected Location:</b></td><td>Page {best_match['page']}</td></tr>
            <tr><td style='padding: 4px 0;'><b>Confidence Score:</b></td><td><span style='background-color: #4caf50; color: white; padding: 2px 6px; border-radius: 4px;'>{score:.4f}</span></td></tr>
        </table>
    </div>
    """

    preview_text = best_match['text']
    pdf_file_path = best_match['pdf_path']  # Active file path to be injected into the viewer

    return details_html, pdf_file_path, f"Page {best_match['page']}", preview_text

# ==============================================================================
# 5. GRADIO GRAPHICAL USER INTERFACE (WEB VIEW UI)
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft(), title="Semantic Search Engine") as demo:
    gr.Markdown("# Cross-Domain Semantic Document Search Engine")
    gr.Markdown("Search your document catalog conceptually. The system interprets meaning, extracts relevant text snippets, and visualizes the target source file instantly.")

    with gr.Row():
        # Left Panel: Controls and Text outputs
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Search Input Query",
                placeholder="Type conceptual ideas here (e.g., neural network training, sports rule variations, medical therapies)..."
            )
            search_btn = gr.Button("Execute Semantic Search ", variant="primary")

            meta_output = gr.HTML(label="Document Metadata Details")
            page_output = gr.Textbox(label="Target Document Page Number", interactive=False)
            preview_output = gr.Textbox(label="Extracted Context Snippet (Paragraph Preview)", lines=6, interactive=False)

        # Right Panel: Live Document Preview Layout
        with gr.Column(scale=1):
            gr.Markdown("### Real-time Embedded Document View")
            pdf_output = gr.File(label="Suggested Source PDF File")

    # Map the click action to execute the search logic
    search_btn.click(
        fn=search_and_display,
        inputs=query_input,
        outputs=[meta_output, pdf_output, page_output, preview_output]
    )

# ==============================================================================
# 6. APP DEPLOYMENT (GENERATES INLINE & PUBLIC REFS)
# ==============================================================================
# Setting share=True gives you a 72-hour public hosting web link!
demo.launch(share=True, debug=True)
