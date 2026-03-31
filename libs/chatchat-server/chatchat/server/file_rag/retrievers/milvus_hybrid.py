"""
替换原 EnsembleRetrieverService 中的 内存bm25方案
改造原因：
旧方案：将所有向量数据拉入 Python 内存，用 jieba 现场构建 BM25 倒排索引，OOM 必崩
新方案：Milvus 原生支持 sparse+dense 混合检索，BM25 索引在数据库底层 C++ 构建，
查询时一次网络请求返回混合排序结果，彻底解放应用层内存
使用前置条件：
  - Milvus >= 2.5.0
  - pymilvus >= 2.5.0


"""

from __future__ import annotations
from typing import List, Optional
from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from chatchat.server.file_rag.retrievers.base import BaseRetrieverService
from chatchat.utils import build_logger

logger = build_logger()

class MilvusHybridRetrieverService(BaseRetrieverService):
    def do_init(
        self,
        collection_name: str = None,
        connection_args: dict = None,
        embedding_function=None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        search_kwargs: dict = None, # === 新增：支持接收底层的过滤条件传参 ===
        ):
        #把所有的必要的配置信息保存成对象属性，供其他方式使用
        self.collection_name = collection_name
        self.connection_args = connection_args
        self.embedding_function = embedding_function
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.search_kwargs = search_kwargs or {}
    @staticmethod
    def from_vectorstore(
        vectorestore: VectorStore,
        top_k : int,
        score_threshold : int | float,
        **kwargs,
    ):
        """
        参数说明：
          vectorstore: Langchain 封装的 Milvus 对象，里面有连接信息、embedding 函数等
          top_k: 最终返回多少条检索结果
          score_threshold: 最低相关度阈值（0~1之间），低于这个值的文档会被过滤掉

        注意：Langchain 社区版的 Milvus 对象把连接参数存在私有属性中，
              无法用 .connection_args 直接取到，所以我们直接从全局配置读取。
        """
        # ★ 关键修复：直接从 Settings 全局配置中读取 Milvus 连接参数
        # 这比从 Langchain 的 Milvus 对象里"挖"要稳定得多
        from chatchat.settings import Settings
        connection_args = Settings.kb_settings.kbs_config.get("milvus")

        # embedding 函数和 collection 名称从 vectorstore 对象取（这两个是可以取到的）
        embedding_function = vectorestore.embedding_func
        collection_name = vectorestore.collection_name

        return MilvusHybridRetrieverService(
            collection_name = collection_name,
            connection_args = connection_args,
            embedding_function = embedding_function,
            top_k = top_k,
            score_threshold = score_threshold,
            search_kwargs=kwargs.get("search_kwargs", {})
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        混合检索的核心执行函数
        当一个用户的问题传进来时，这个函数去数据库中查相关的chunk
        1.用embedding 模型把 query 转换成向量
        2.发给Milvus，让它同时进行向量检索和关键词检索
        3.milvus内部用RRF 算法融合两路的结果并排序
        4.Milvus 格式数据转换成Langchain 的Document对象返回。
        """
        try:
            from pymilvus import(
                MilvusClient,
                AnnSearchRequest,   # 向量近邻检索请求
                RRFRanker,          # RRF
            )
        except ImportError:
            raise ImportError(
                "使用 Milvus 混合检索需要 pymilvus >= 2.5.0，"
                "请执行：pip install pymilvus>=2.5.0"
            )
        # 1. 建立连接：通过pymilvus的原生客户端直接连接Milvus数据库
        # 不用Langchain的封装，直接使用pymilvus 底层 API来发送请求
        client = MilvusClient(
            uri=f"http://{self.connection_args.get('host', '127.0.0.1')}:{self.connection_args.get('port', '19530')}",
            token=self.connection_args.get("user", "") + ":" + self.connection_args.get("password", ""),
        )

        # 2. 把用户的文字问题通过embedding 模型转换为向量
        # 这里的embedding模型是配置文件中的模型
        logger.info(f"[混合检索] query进行embedding: {query[:50]}...")
        query_dense_vector = self.embedding_function.embed_query(query)

        # 提取外界传进来的过滤条件
        expr = self.search_kwargs.get("expr", None)

        # 3. 构建向量检索请求（语义检索）
        # anns_field: Milvus collection 里存向量的字段名，默认是 vector
        # metric_type: 向量距离计算公式 IP：内积，L2=欧氏距离
        # 候选数量用top_k * 5 是因为两路结果可能有重叠，多一点候选文档效果会更好
        dense_search_request = AnnSearchRequest(
            data=[query_dense_vector],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"ef": 200}},
            limit=self.top_k * 5,  # 多取一些，最后RRF再裁减到top_k
            expr=expr, # === 这里透传过滤条件 ===
        )

        # 4. 关键字检索（BM25 稀疏检索）
        # "sparse_vector" 字段是在写数据时，Milvus的BM25BuiltInFunction自动建的稀疏向量列
        # 此处的 data 我们传入原始文本，让 Milvus 引擎内部做 BM25 分词和权重计算
        sparse_search_request = AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=self.top_k * 5,
            expr=expr, # === 这里透传过滤条件 ===
        )

        # 5. 使用 RRF 融合两路检索结果
        # RRFRanker 是 Reciprocal Rank Fusion 算法的实现
        # 核心思想：一篇文档在两路检索中排名都靠前，它的综合分就很高
        # k 是平滑参数，官方推荐值为 60
        rrf_ranker = RRFRanker(k=60)

        # 6. 执行混合检索：一次网络请求，两路结果同时返回
        logger.info("[混合检索] 正在向 Milvus 发起 hybrid_search 请求...")
        results = client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_search_request, sparse_search_request],  # 发送两路请求
            ranker=rrf_ranker,
            limit=self.top_k,
            # === 将新增的 parent_id 和 is_parent 也一并从数据库取出来（之前写死了只取 text/source/pk，导致报错拿不到 None） ===
            output_fields=["text", "source", "pk", "parent_id", "is_parent"],  
        )

        # 7. 把Milvus返回的结果转换成Langchain 的Document对象
        # Langchain 的 Document 对象有 page_content（文本内容）和 metadata（元数据）两个字段
        documents = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                text = entity.get("text", "")

                # 构建metadata，把source（文件来源）等信息保存下来
                metadata = {k: v for k, v in entity.items() if k != "text"}
                metadata["score"] = hit.get("distance", 0.0)  # 把综合得分也存进去

                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)

        logger.info(f"[混合检索] 检索完成，共返回 {len(documents)} 条文档")
        return documents