from chatchat.server.file_rag.retrievers import (
    BaseRetrieverService,
    EnsembleRetrieverService,
    VectorstoreRetrieverService,
    MilvusVectorstoreRetrieverService,
    MilvusHybridRetrieverService,
)

Retrivals = {
    "milvusvectorstore": MilvusVectorstoreRetrieverService,
    "vectorstore": VectorstoreRetrieverService,
    "ensemble": EnsembleRetrieverService,
    "milvushybrid": MilvusHybridRetrieverService,
}


def get_Retriever(type: str = "vectorstore") -> BaseRetrieverService:
    return Retrivals[type]
