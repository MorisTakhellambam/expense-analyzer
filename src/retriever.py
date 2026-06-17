from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DB_PATH

# load the same embedding model as in embedding.py
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}        # required for cosine similarity
)

# connect to the persisted ChromaDB collection
vector_store = Chroma(
    collection_name="expenses",
    embedding_function=embedding_model,
    persist_directory=CHROMA_DB_PATH,
    collection_metadata={"hnsw:space": "cosine"}        # ensure the same distance metric is used
)

base_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}      # retrieve top 10 most similar expenses
)

def get_retriever(category: str = None, date: str = None, k: int = 10):
    """
    Returns a retriever with optional metadata filters.

    Examples:
        get_retriever()  # retrieves top 10 most similar expenses without filters
        get_retriever(category="Food")  # retrieves top 10 most similar expenses in the "Food" category
        get_retriever(category="Food", day="Saturday")  # retrieves top 10 most similar expenses in the "Food" category on "01-01-2026"
        get_retriever(date="01-01-2026")  # retrieves top 10 most similar expenses on "01-01-2026"
    """

    where = _build_where(category, date)

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            **({"filter": where} if where else {})
        },
    )

def _build_where(category: str = None, date: str = None):
    """
    Builds a ChromaDB $and/$or filter dict from optional fields.
    """

    conditions = []

    if category:
        conditions.append({"category": {"$eq": category.title()}})  # ensure category is title-cased to match stored metadata
    
    if date:
        conditions.append({"date": {"$eq": date}})  # ensure date matches stored metadata

    if len(conditions) == 0:
        return None  # no filters, retrieve all
    
    if len(conditions) == 1:
        return conditions[0]  # single filter, no need for $and
    
    return {"$and": conditions}  # multiple filters, combine with $and

def query(question: str, category: str = None, date: str = None, k: int = 10    ):
    """
    Run a semantic search and return matching expense strings.
    
    Usage:
        query("restaurants and dining")
        query("transport", date="01-01-2026")
        query("big purchases", category="Shopping", k=5)
    """

    retriever = get_retriever(category=category, date=date, k=k)
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]