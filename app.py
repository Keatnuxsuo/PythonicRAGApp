import os
from typing import List, Dict, Any
from chainlit.types import AskFileResponse
from aimakerspace.text_utils import CharacterTextSplitter, TextFileLoader, PDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
import chainlit as cl

# Initialize Qdrant client
qdrant_client = QdrantClient(":memory:")  # For development, use in-memory storage
# For production, you might want to use:
# qdrant_client = QdrantClient(url="http://localhost:6333")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Initialize LLM
llm = ChatOpenAI(temperature=0)

# Create the prompt template
template = """Use the following context to answer the question. If you cannot find the answer in the context, say you don't know the answer.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

text_splitter = CharacterTextSplitter()

def process_file(file: AskFileResponse):
    import tempfile
    import shutil
    
    print(f"Processing file: {file.name}")
    
    # Create a temporary file with the correct extension
    suffix = f".{file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        # Copy the uploaded file content to the temporary file
        shutil.copyfile(file.path, temp_file.name)
        print(f"Created temporary file at: {temp_file.name}")
        
        # Create appropriate loader
        if file.name.lower().endswith('.pdf'):
            loader = PDFLoader(temp_file.name)
        else:
            loader = TextFileLoader(temp_file.name)
            
        try:
            # Load and process the documents
            documents = loader.load_documents()
            texts = text_splitter.split_texts(documents)
            return texts
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file.name)
            except OSError as e:
                print(f"Error cleaning up temporary file: {e}")

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a Text or PDF file to begin!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=2,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`..."
    )
    await msg.send()

    # Process the file
    texts = process_file(file)
    print(f"Processing {len(texts)} text chunks")

    # Create Qdrant vector store
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name="documents",
        embeddings=embeddings,
    )
    
    # Add documents to vector store
    vector_store.add_texts(texts)
    
    # Create the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Create the RAG chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Store the chain in the user session
    cl.user_session.set("chain", rag_chain)
    
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    
    msg = cl.Message(content="")
    await msg.send()
    
    # Stream the response
    async for chunk in chain.astream(message.content):
        await msg.stream_token(chunk)
    
    await msg.update()
