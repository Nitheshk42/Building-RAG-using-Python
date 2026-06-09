from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,      # Smaller chunks
        chunk_overlap=100,
        separators=["\n\n", "\n", "•", "-", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    
    # Add section headers for better matching
    enhanced = []
    for split in splits:
        content = split.page_content
        if any(x in content.lower() for x in ["name", "email", "phone"]):
            content = "[PERSONAL]\n" + content
        elif any(x in content.lower() for x in ["experience", "worked", "developed"]):
            content = "[EXPERIENCE]\n" + content
        elif any(x in content.lower() for x in ["skill", "proficient"]):
            content = "[SKILLS]\n" + content
        
        split.page_content = content
        enhanced.append(split)
    
    return enhanced