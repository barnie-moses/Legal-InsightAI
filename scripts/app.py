import os
import shutil
import requests
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


# Download PDF if missing
def download_pdf(url, path):
    if not os.path.exists(path):
        st.info("Downloading PDF...")
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)
        st.success("Download complete!")


# Load and Chunk Document
def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    return splitter.split_documents(raw_docs)


# Setup Vector DB
def create_vector_store(documents, embedding_model, vector_db_dir):
    if os.path.exists(vector_db_dir):
        try:
            return FAISS.load_local(vector_db_dir, embeddings=embedding_model, allow_dangerous_deserialization=True)
        except AssertionError:
            shutil.rmtree(vector_db_dir)
    vector_db = FAISS.from_documents(documents, embedding_model)
    vector_db.save_local(vector_db_dir)
    return vector_db


# Setup LLM QA Chain
def setup_chain(vector_db):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
As a legal reference assistant, analyze the provided CONTEXT from the National Archives laws document
and provide the most precise answer to the QUESTION.

RULES:
1. MUST answer using ONLY the CONTEXT provided
2. Cite exact section numbers (e.g., '44 U.S.C. § 2102') when available
3. If uncertain, say "The document does not specify"
4. For definitions: provide the exact quoted definition
5. For procedures: list steps in order

CONTEXT:
{context}

QUESTION:
{question}

STRUCTURED RESPONSE:
[Summary Answer]
[Relevant Section]
[Direct Quote (if applicable)]
"""
    )

    llm = Ollama(
        model="llama3",
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.1
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    combine_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    return RetrievalQA(retriever=retriever, combine_documents_chain=combine_chain, return_source_documents=True)


# Ask a Question
def ask_question(chain, question, write_to_file=False):
    response = chain.invoke({"query": question})
    answer = response["result"]
    sources = response["source_documents"]

    # Write to file
    if write_to_file:
        with open("/home/moses/Moses/Legal-InsightAI/data/responses.txt", "a") as f:
            f.write(f"QUESTION: {question}\n")
            f.write(f"ANSWER: {answer}\n")
            for i, doc in enumerate(sources[:2]):
                page = doc.metadata.get('page', 'N/A')
                f.write(f"\nSource {i+1} (Page {page}):\n{doc.page_content[:500]}...\n")
            f.write("\n" + "=" * 80 + "\n\n")

    return answer, sources


# Streamlit UI
def main():
    st.set_page_config(page_title="Legal Agent (NARA Laws)", layout="wide")
    st.title("National Archives and Records Administration (NARA) Law Assistant")
    st.markdown("Ask questions about the **Basic Laws and Authorities of the National Archives (2016)**")

    # Setup
    PDF_URL = "https://www.archives.gov/files/about/laws/basic-laws-book-2016.pdf"
    PDF_PATH = "/home/moses/Moses/Legal-InsightAI/data/basic_laws_2016.pdf"
    VECTOR_DB_DIR = "/home/moses/Moses/Legal-InsightAI/database/faiss_basic_laws"

    download_pdf(PDF_URL, PDF_PATH)
    documents = load_documents(PDF_PATH)

    embedding = HuggingFaceInstructEmbeddings(
        model_name="nlpaueb/legal-bert-base-uncased",
        query_instruction="Represent this legal question for searching legislation:"
    )

    vector_db = create_vector_store(documents, embedding, VECTOR_DB_DIR)
    chain = setup_chain(vector_db)

    # Input box
    question = st.text_input("Ask your legal question:", placeholder="E.g. What are the Archivist's duties?")
    if question:
        with st.spinner("Searching and generating answer..."):
            answer, sources = ask_question(chain, question)

            st.subheader("Answer")
            st.markdown(
                f"""
                <div style='
                    padding: 1rem;
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    font-size: 1.2rem;
                    line-height: 1.6;
                    color: #333;
                '>
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("Top Sources"):
                for i, doc in enumerate(sources[:2]):
                    page = doc.metadata.get("page", "N/A")
                    st.markdown(
                        f"<div style='font-size: 1.1rem; font-weight: bold;'>Source {i+1} (Page {page})</div>",
                        unsafe_allow_html=True
                    )
                    st.code(doc.page_content[:500] + "...")

    st.markdown("---")
    st.caption(
        "This assistant is for informational purposes only and does not provide legal advice. "
        "The content is based on publicly available documents acquired via open web search, "
        "specifically the 2016 edition of 'Basic Laws and Authorities of the National Archives and Records Administration'. "
        "The developer does not claim legal authority or ownership over any material referenced. "
        "Powered by LangChain · Ollama · HuggingFace · Streamlit"
    )

if __name__ == "__main__":
    main()
