from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

def get_rag_chain(vectorstore):
    """Create RAG chain for StudySage"""
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Good system prompt for study assistant
    template = """You are StudySage, a helpful and accurate AI study assistant.
    Answer the question based only on the provided context.
    If you don't know the answer, say "I don't have enough information in the documents."
    
    Context: {context}
    
    Question: {question}
    
    Answer in a clear, student-friendly way:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # For source citations
    def rag_with_sources(query):
        docs = retriever.invoke(query)
        context = format_docs(docs)
        answer = rag_chain.invoke(query)
        return {"answer": answer, "context": docs}
    
    return rag_with_sources