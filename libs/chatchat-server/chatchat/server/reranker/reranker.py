import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from typing import Any, List, Optional, Sequence

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from pydantic.v1 import Field, PrivateAttr
from sentence_transformers import CrossEncoder


class LangchainReranker(BaseDocumentCompressor):
    """Document compressor that uses `Cohere Rerank API`."""

    model_name_or_path: str = Field()
    _model: Any = PrivateAttr()
    top_n: int = Field()
    device: str = Field()
    max_length: int = Field()
    batch_size: int = Field()
    # show_progress_bar: bool = None
    num_workers: int = Field()

    # activation_fct = None
    # apply_softmax = False

    def __init__(
        self,
        model_name_or_path: str,
        top_n: int = 3,
        device: str = "cuda",
        max_length: int = 1024,
        batch_size: int = 32,
        # show_progress_bar: bool = None,
        num_workers: int = 0,
        # activation_fct = None,
        # apply_softmax = False,
    ):
        # self.top_n=top_n
        # self.model_name_or_path=model_name_or_path
        # self.device=device
        # self.max_length=max_length
        # self.batch_size=batch_size
        # self.show_progress_bar=show_progress_bar
        # self.num_workers=num_workers
        # self.activation_fct=activation_fct
        # self.apply_softmax=apply_softmax
# 初始化：加载 Cross-Encoder 模型
# 这里调用了 NLP 界大名鼎鼎的 sentence-transformers 库的 CrossEncoder 类-
# 在这个瞬间，机器把几个 G 的 Reranker 模型（比如 BAAI 的 bge-reranker）从磁盘生生拽进了内存或 GPU (device="cuda") 中。
# 注意参数 max_length，它在强行规定专家审阅每一篇文章的最长字数，防止内存被撑爆。
        self._model = CrossEncoder(
            model_name_or_path=model_name_or_path, max_length=max_length, device=device
        )
        super().__init__(
            top_n=top_n,
            model_name_or_path=model_name_or_path,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            # show_progress_bar=show_progress_bar,
            num_workers=num_workers,
            # activation_fct=activation_fct,
            # apply_softmax=apply_softmax
        )
    # 核心方法：接收候选文档列表 + 用户问题 → 返回精排后的 Top-N
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Cohere's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        # 把每个 doc 和 query 拼成 [query, doc] 对
        sentence_pairs = [[query, _doc] for _doc in _docs]
        # 送入 Cross-Encoder，得到每对的相关性分数
        results = self._model.predict(
            sentences=sentence_pairs,
            batch_size=self.batch_size,
            #  apply_softmax=self.apply_softmax,
            convert_to_tensor=True,
        )
        top_k = self.top_n if self.top_n < len(results) else len(results)
        # 取分数最高的 top_n 个文档返回
        values, indices = results.topk(top_k)
        final_results = []
        for value, index in zip(values, indices):
            doc = doc_list[index]
            doc.metadata["relevance_score"] = value
            final_results.append(doc)
        return final_results


# if __name__ == "__main__":
    # 不再适用
    # from chatchat.configs import (
    #     MODEL_PATH,
    #     RERANKER_MAX_LENGTH,
    #     RERANKER_MODEL,
    #     SCORE_THRESHOLD,
    #     TEMPERATURE,
    #     USE_RERANKER,
    #     VECTOR_SEARCH_TOP_K,
    # )

    # if USE_RERANKER:
    #     reranker_model_path = MODEL_PATH["reranker"].get(
    #         RERANKER_MODEL, "BAAI/bge-reranker-large"
    #     )
    #     print("-----------------model path------------------")
    #     print(reranker_model_path)
    #     reranker_model = LangchainReranker(
    #         top_n=3,
    #         device="cpu",
    #         max_length=RERANKER_MAX_LENGTH,
    #         model_name_or_path=reranker_model_path,
    #     )
