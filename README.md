# Volvo XC90 2027 – RAG-Based AI Chatbot

An intelligent **Retrieval-Augmented Generation (RAG) chatbot** designed to answer user queries based on the official Volvo XC90 2027 User Manual.

This project combines **Pinecone Vector Database, Hugging Face Embeddings, LangChain, Ollama (Llama 3.2), and Gradio** to provide context-aware answers by retrieving relevant information from the vehicle's user manual.

---

## Project Overview

Traditional chatbots may generate answers using general knowledge, which can sometimes result in inaccurate or irrelevant information.

To address this limitation, this project implements a Retrieval-Augmented Generation (RAG) architecture. It retrieves relevant text from the Volvo XC90 2027 User Manual and provides it as context to a Large Language Model (LLM), which then generates an answer based on the retrieved information.

The chatbot is designed to answer questions related to vehicle features, safety systems, operating instructions, and other information covered in the manual.

## Project Objectives

- Build a document-based question-answering system using RAG.
- Extract and process information from a PDF user manual.
- Generate semantic embeddings using Hugging Face.
- Store and retrieve document embeddings using Pinecone.
- Integrate an LLM through Ollama for natural-language responses.
- Develop an interactive chatbot interface using Gradio.
- Reduce unsupported responses by grounding answers in retrieved document context.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG pipeline and prompt management |
| Pinecone | Vector database and similarity search |
| Hugging Face | Text embedding generation |
| all-MiniLM-L6-v2 | Sentence embedding model |
| Ollama | Local LLM execution |
| Llama 3.2 | Response generation |
| Gradio | Interactive chatbot interface |
| PyPDF | PDF document processing |
| RecursiveCharacterTextSplitter | Document chunking |

## System Architecture

The project follows a Retrieval-Augmented Generation workflow.

**Document Processing and Indexing**

1. Load the Volvo XC90 2027 User Manual PDF.
2. Extract text from the document.
3. Divide the extracted text into smaller chunks.
4. Generate vector embeddings using `all-MiniLM-L6-v2`.
5. Store the embeddings and associated metadata in Pinecone.

**Question Answering**

1. The user enters a question through the Gradio interface.
2. The question is converted into a vector embedding.
3. Pinecone performs similarity search to retrieve relevant document chunks.
4. The retrieved context is passed to the Llama 3.2 model through Ollama.
5. The LLM generates a context-based response.
6. The generated answer is displayed through the Gradio interface.

## RAG Workflow

```
        Volvo XC90 2027 PDF
                 |
                 v
          Text Extraction
                 |
                 v
          Text Chunking
                 |
                 v
       Hugging Face Embeddings
                 |
                 v
          Pinecone Database
                 |
                 |
       User Enters a Question
                 |
                 v
        Generate Query Embedding
                 |
                 v
         Pinecone Similarity Search
                 |
                 v
        Retrieve Relevant Context
                 |
                 v
          Ollama - Llama 3.2
                 |
                 v
         Context-Based Answer
                 |
                 v
          Gradio Chatbot UI
```

## Dataset and Document Processing

**Source Document:** Volvo XC90 2027 User Manual (PDF)

Document processing details:

| Parameter | Value |
|---|---|
| Document pages | 443 |
| Total text chunks | 1,060 |
| Chunk size | 1,000 characters |
| Chunk overlap | 200 characters |
| Embedding model | all-MiniLM-L6-v2 |
| Embedding dimension | 384 |
| Vector database | Pinecone |
| Pinecone index | volvo2 |
| Namespace | volvo_xc90_manual_v1 |
| Retrieval method | Similarity Search |

## Project Structure

```text
Volvo-XC90-RAG-Chatbot/
│
├── pinecone_app.py
│
├── pinecone_iteration.ipynb
│
├── requirements.txt
│
├── .env.example
│
├── .gitignore
│
├── README.md
│
└── screenshots/
    ├── 01_airbag_information.png
    ├── 02_infotainment_instructions.png
    ├── 03_child_safety_information.png
    ├── 04_out_of_context_question.png
    └── 05_airbag_comparison.png
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kulpreetkaurs/Volvo-XC90-RAG-Chatbot.git
```

```bash
cd Volvo-XC90-RAG-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv pinecone_env
```

Activate it on Windows:

```powershell
.\pinecone_env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Pinecone API Key

Create a Pinecone account and obtain your API key.

Create a `.env` file in the project directory:

```env
PINECONE_API_KEY=your_pinecone_api_key
```

Never upload your actual `.env` file or API key to GitHub.

### 5. Configure Ollama

Install Ollama from:

https://ollama.com/

Download the Llama 3.2 model:

```bash
ollama pull llama3.2
```

Ensure Ollama is running locally before starting the application.

### 6. Configure Pinecone

Create a Pinecone index with the following configuration:

- Index name: `volvo2`
- Dimension: `384`
- Metric: `cosine`

Ensure that the index contains the manual's document vectors in the namespace:

`volvo_xc90_manual_v1`

The vector database must be populated before running the chatbot. The notebook `pinecone_iteration.ipynb` contains the document-processing and indexing workflow.

### 7. Run the application

```bash
python pinecone_app.py
```

Open the local Gradio URL displayed in the terminal, usually:

```text
http://127.0.0.1:7860
```

## 🖥️ Application Screenshots

### 1. Airbag Information

Ask questions about the different types of airbags and their functions.

![Airbag Information](screenshots/01_airbag_information.png)

### 2. Infotainment Instructions

Retrieve instructions related to the vehicle's infotainment system.

![Infotainment Instructions](screenshots/02_infotainment_instructions.png)

### 3. Child Safety Information

Retrieve safety-related information about child restraints.

![Child Safety](screenshots/03_child_safety_information.png)

### 4. Out-of-Context Question

Test whether the chatbot avoids answering questions that are unrelated to the manual.

![Out of Context Question](screenshots/04_out_of_context_question.png)

### 5. Comparative Question

Compare different vehicle safety features using retrieved document information.

![Airbag Comparison](screenshots/05_airbag_comparison.png)

## Key Features

- **Document-Based Question Answering:** Answers questions using information retrieved from the Volvo XC90 2027 User Manual.
- **Semantic Search:** Retrieves relevant document chunks using vector similarity.
- **Context-Aware Responses:** Uses retrieved information as context for LLM-generated answers.
- **Hallucination Control:** Uses a context-restricted prompt to discourage unsupported responses.
- **Local LLM Integration:** Runs Llama 3.2 through Ollama.
- **Interactive Interface:** Provides an easy-to-use Gradio chatbot.
- **Scalable Vector Storage:** Uses Pinecone to store and retrieve document embeddings.

## Future Enhancements

- Add support for multiple vehicle manuals and PDF documents.
- Implement conversation history and multi-turn question answering.
- Improve retrieval using hybrid search and reranking.
- Add source citations and clickable PDF page references.
- Deploy the application using a cloud-hosted LLM and suitable vector database.
- Improve the interface with a conversational chat layout.

## Limitations

- The chatbot is currently designed around the Volvo XC90 2027 User Manual.
- Answer quality depends on the relevance and completeness of retrieved document chunks.
- The LLM runs locally through Ollama, so the application requires Ollama and the specified model.
- Pinecone API credentials and an appropriately populated Pinecone index are required.
- Responses should be verified against the official vehicle manual, particularly for safety-critical instructions.

## Author

**Kulpreet Kaur Sembhi**

BCA Graduate | Aspiring Data Scientist | AI & Machine Learning Enthusiast

**Technical Interests:** Data Science, Machine Learning, Generative AI, Retrieval-Augmented Generation (RAG), Python, SQL and Data Analytics.

- GitHub: https://github.com/kulpreetkaurs

---

⭐ If you find this project interesting, consider giving the repository a star!
