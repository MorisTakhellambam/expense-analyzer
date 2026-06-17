import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CLEANED_DATA_PATH_WITH_TEXT

from sentence_transformers import SentenceTransformer
import chromadb

def create_embeddings() -> None:
    """
    Creates embeddings for the text data in the cleaned DataFrame and saves them to a ChromaDB collection.
    """
    # Load the cleaned data with text
    df = pd.read_csv(CLEANED_DATA_PATH_WITH_TEXT)
    
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for the 'text' column
    embeddings = model.encode(
        df["text"].tolist(),        # ChromaDB expects plain lists, not tensors
        batch_size=64,
        show_progress_bar=True 
    )

    chroma = chromadb.PersistentClient(path="./data/expense_vector_db")
    collection = chroma.get_or_create_collection(
        name="expenses",
        metadata={"hnsw:space": "cosine"}  # MiniLM was trained with cosine similarity
    )

    collection.add(
        ids        = [str(i) for i in df.index],
        documents  = df["text"].tolist(),
        embeddings = embeddings,
        metadatas  = [
            {
                "date"        : row["date"],
                "category"    : row["category"],
                "amount"      : row["amount"],
                "description" : row["description"],
            }
            for _, row in df.iterrows()
        ]
    )

    print(f"Indexed {len(df)} expenses into ChromaDB")

if __name__ == "__main__":
    create_embeddings()