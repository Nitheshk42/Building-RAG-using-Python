from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pathlib import Path
from src.llm_provider import get_llm
from src.rag_pipeline_hybrid import _retrieval_query

# Load from root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)


def get_rag_chain(vectorstore):
    # k=3 was the root cause of "I don't have that information" for things that ARE on the
    # resume (e.g. "what is your recent project and timeline") - too few chunks retrieved to
    # reliably include the specific project/date info. Bumped to match the other chains.
    llm = get_llm(temperature=0.3, max_tokens=2000)

    template = """You are StudySage, answering interview-prep questions grounded in the
candidate's resume below.

SPECIFICITY RULE: If the resume context contains concrete numbers, metrics, config values,
tool versions, company names, or dates, quote them verbatim rather than paraphrasing into
generic statements - concrete detail is what makes an answer credible.

RECENCY RULE: If asked about the "recent," "current," "latest," or "most recent" project/role,
scan ALL project/role entries in the context, compare their date ranges, and answer about the
one that's actually most recent (marked "Current"/"Present", or the latest start date) - do not
just answer about whichever chunk happens to appear first.

If the resume context genuinely doesn't cover something, say so plainly rather than guessing.

RESUME CONTEXT:
{context}

QUESTION: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(_retrieval_query(x["question"]))),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain