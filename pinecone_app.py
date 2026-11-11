
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import time
import ollama
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found. Check your .env file.")


# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "volvo2"
namespace = "volvo_xc90_manual_v1"

index = pc.Index(index_name)


# Connect to Ollama LLM
def call_llm(
    prompt: str,
    temperature: float = 0,
    max_output_tokens: int = 2048,
    retries: int = 3
) -> str:

    for attempt in range(retries):
        try:
            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": temperature,
                    "num_predict": max_output_tokens
                }
            )

            return response["message"]["content"]

        except Exception as e:

            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"[retry {attempt + 1}] {e} - waiting {wait}s")
                time.sleep(wait)

            else:
                raise


# Retrieve relevant documents from Pinecone
def retrieve_context(question, top_k=5):

    query_embedding = embedding_model.embed_query(question)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    context = []

    for match in results.matches:

        metadata = match.metadata or {}

        text = metadata.get("text", "")

        if text:
            page = metadata.get("page_label", "Unknown")

            context.append(
                f"[PDF Page {page}]\n{text}"
            )

    return context


# Generate answer using retrieved context
def ask_question(question):

    if not question.strip():
        return "Please enter a question."

    context = retrieve_context(question)

    if not context:
        return "No relevant information was retrieved from the manual."

    context_str = "\n\n".join(context)

    prompt = f"""
You are a helpful assistant specialised in the
Volvo XC90 2027 user manual.

Answer the user's question using ONLY the
information provided in the context.

Instructions:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. Include all relevant details available in the context.
4. If information is not available, clearly say:
   "This information is not available in the
   provided Volvo XC90 2027 manual."
5. Give clear and easy-to-understand answers.
6. Mention relevant PDF page numbers wherever possible.

Context:
{context_str}

Question:
{question}

Answer:
"""

    answer = call_llm(prompt)

    return answer


# Gradio user interface
demo = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask a question about the Volvo XC90 2027 manual..."
    ),
    outputs=gr.Textbox(
        label="Answer",
        lines=10
    ),
    title="Volvo XC90 2027 RAG Chatbot",
    description=(
        "Ask questions about the Volvo XC90 2027 user manual "
        "and get answers based on information retrieved "
        "from the Pinecone vector database."
    ),
    theme="default"
)


# Launch application
demo.launch(debug=True, share=True)
