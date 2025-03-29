import os
from langchain_community.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_github_repo_and_create_faiss_index(github_url, repo_path="./temp_repo", 
                                           chunk_size=1000, chunk_overlap=200):
    """
    Load code from a GitHub repository, split into chunks, and create a FAISS index.
    """
    # Clone the repository
    print(f"Cloning repository from {github_url}...")
    loader = GitLoader(
        clone_url=github_url,
        repo_path=repo_path,
        branch="main"
    )
    
    # Load documents
    print("Loading documents...")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings and FAISS index
    print("Creating embeddings and FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.from_documents(chunks, embeddings)
    
    print("FAISS index created successfully")
    return faiss_index

def search_repository(faiss_index, query, k=5):
    """
    Search the FAISS index for similar chunks.
    """
    results = faiss_index.similarity_search_with_score(query, k=k)
    return results

if __name__ == "__main__":
    # Example usage
    github_url = "https://github.com/viarotel-org/escrcpy.git"
    # Load repository and create FAISS index
    faiss_index = load_github_repo_and_create_faiss_index(github_url)
    
    # Example search
    query = "example search query"
    results = search_repository(faiss_index, query)
    
    # Print results
    print(f"\nSearch results for query: '{query}'")
    for doc, score in results:
        print(f"Score: {score}")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print("="*80)
    
    # Save the FAISS index for later use
    faiss_index.save_local("faiss_index")
    print("FAISS index saved to 'faiss_index' directory")