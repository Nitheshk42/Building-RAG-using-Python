from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents):
    # Create intro chunk
    intro = Document(
        page_content="[PROFILE] This is Nithesh Kumar's resume. Contact: chavan3888@gmail.com. Professional Software Engineer with 4+ years experience.",
        metadata={"source": "intro"}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", "•", "-", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    
    # Add section headers
    enhanced = [intro]
    for split in splits:
        content = split.page_content
        if "experience" in content.lower() or "worked" in content.lower():
            content = "[EXPERIENCE]\n" + content
        split.page_content = content
        enhanced.append(split)
    
    return enhanced