from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    """Chunk the uploaded resume. No fabricated/hardcoded identity text is injected -
    every chunk comes directly from the actual uploaded document."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", "•", "-", " ", ""]
    )

    splits = text_splitter.split_documents(documents)

    enhanced = []
    for split in splits:
        content = split.page_content
        if "experience" in content.lower() or "worked" in content.lower():
            content = "[EXPERIENCE]\n" + content
        split.page_content = content
        enhanced.append(split)

    return enhanced