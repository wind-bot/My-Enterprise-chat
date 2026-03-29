import os
from typing import Dict, List, Optional

from langchain.schema import Document
from langchain.vectorstores.milvus import Milvus

from chatchat.settings import Settings
from chatchat.server.db.repository import list_file_num_docs_id_by_kb_name_and_file_name
from chatchat.server.utils import get_Embeddings
from chatchat.server.file_rag.utils import get_Retriever
from chatchat.server.knowledge_base.kb_service.base import (
    KBService,
    SupportedVSType,
    score_threshold_process,
)
from chatchat.server.knowledge_base.utils import KnowledgeFile


class MilvusKBService(KBService):
    milvus: Milvus

    @staticmethod
    def get_collection(milvus_name):
        from pymilvus import Collection

        return Collection(milvus_name)

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        result = []
        if self.milvus.col:
            # ids = [int(id) for id in ids]  # for milvus if needed #pr 2725
            data_list = self.milvus.col.query(
                expr=f"pk in {[int(_id) for _id in ids]}", output_fields=["*"]
            )
            for data in data_list:
                text = data.pop("text")
                result.append(Document(page_content=text, metadata=data))
        return result

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        self.milvus.col.delete(expr=f"pk in {ids}")

    @staticmethod
    def search(milvus_name, content, limit=3):
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        c = MilvusKBService.get_collection(milvus_name)
        return c.search(
            content, "embeddings", search_params, limit=limit, output_fields=["content"]
        )

    def do_create_kb(self):
        pass

    def vs_type(self) -> str:
        return SupportedVSType.MILVUS

    def _load_milvus(self):
        self.milvus = Milvus(
            embedding_function=get_Embeddings(self.embed_model),
            collection_name=self.kb_name,
            connection_args=Settings.kb_settings.kbs_config.get("milvus"),
            index_params=Settings.kb_settings.kbs_config.get("milvus_kwargs")["index_params"],
            search_params=Settings.kb_settings.kbs_config.get("milvus_kwargs")["search_params"],
            auto_id=True,
            )

    def _ensure_hybrid_collection(self):
        """
        确保 Milvus Collection 里同时存在 Dense 向量字段 和 BM25 稀疏向量字段。

        为什么需要这个方法？
          - Langchain 封装的 Milvus 类在建 Collection 时，只会建 Dense 向量字段（vector 列）
          - 要支持原生 BM25 混合检索，Collection 还必须有 sparse_vector 列
          - 此方法在 do_init 时检查，如果 Collection 不存在或字段不完整，
            就用 pymilvus 的原生 API 从头重建一个"双字段"的 Collection

        需要 Milvus >= 2.5.0 才支持 BM25BuiltInFunction
        """
        from pymilvus import (
            MilvusClient,
            DataType,
            Function,
            FunctionType,
        )
        from chatchat.settings import Settings

        conn = Settings.kb_settings.kbs_config.get("milvus")
        uri = f"http://{conn.get('host', '127.0.0.1')}:{conn.get('port', '19530')}"
        client = MilvusClient(uri=uri)

        col_name = self.kb_name

        # 检查 collection 是否已经存在，且字段是否完整
        if client.has_collection(col_name):
            schema = client.describe_collection(col_name)
            field_names = [f["name"] for f in schema.get("fields", [])]
            if "sparse_vector" in field_names:
                # 已经有稀疏向量字段，直接返回，无需重建
                return
            # 字段不完整，删掉旧的
            client.drop_collection(col_name)

        # ★ 关键：动态探测 Embedding 模型的真实输出维度，不能写死
        # 不同模型维度不一样：bge-m3 = 1024, embedding-3 = 2048, text-embedding-3-large = 3072
        embed_func = get_Embeddings(self.embed_model)
        test_vec = embed_func.embed_query("维度探测")
        actual_dim = len(test_vec)
        print(f"[MilvusKBService] 探测到 Embedding 维度：{actual_dim}")

        # ★ 核心：用 pymilvus 原生 API 手动建 Schema，加入两种向量字段
        schema = client.create_schema()
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535,
                         enable_analyzer=True)                       # 开启分词分析器，供 BM25 使用
        schema.add_field("vector", DataType.FLOAT_VECTOR,
                         dim=actual_dim)                             # Dense 向量（动态维度）
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)  # BM25 稀疏向量
        schema.add_field("source", DataType.VARCHAR, max_length=512,
                         default_value="")                           # 文件来源

        # 定义 BM25 内置函数：自动把 text 字段的文本转换成 sparse_vector 稀疏向量
        bm25_function = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
        )
        schema.add_function(bm25_function)

        # 建索引：Dense 用 HNSW，Sparse 用 SPARSE_INVERTED_INDEX
        index_params = client.prepare_index_params()
        index_params.add_index("vector", index_type="HNSW",
                               metric_type="IP",
                               params={"M": 16, "efConstruction": 200})
        index_params.add_index("sparse_vector",
                               index_type="SPARSE_INVERTED_INDEX",
                               metric_type="BM25")

        client.create_collection(col_name, schema=schema,
                                 index_params=index_params)
        print(f"[MilvusKBService] 已为知识库 '{col_name}' 建立混合检索 Collection（Dense + BM25）")

    def do_init(self):
        self._ensure_hybrid_collection()   # ← 新增：确保双字段 Collection 存在
        self._load_milvus()

    def do_drop_kb(self):
        if self.milvus.col:
            self.milvus.col.release()
            self.milvus.col.drop()

    def do_search(self, query: str, top_k: int, score_threshold: float):
        """
            改造前：只走 Dense 向量检索
            改造后：走 Dense + BM25 Sparse 混合检索
            为什么要在这里改？
            因为 do_search 是知识库被调用时的入口函数，
            用户问一个问题 → 调用 do_search → 返回相关文档 → 这些文档被拼到 Prompt 里给 LLM
            只要改这一个函数，整个知识库的检索链路就全部升级了
    """
        self._load_milvus()
        # embed_func = get_Embeddings(self.embed_model)
        # embeddings = embed_func.embed_query(query)
        # docs = self.milvus.similarity_search_with_score_by_vector(embeddings, top_k)
        retriever = get_Retriever("milvushybrid").from_vectorstore(
            self.milvus,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        docs = retriever.get_relevant_documents(query)
        return docs

    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        """
        改造后：绕过 Langchain 封装，直接用 pymilvus 原生 API 写入数据

        为什么要绕过 Langchain？
          Langchain 的 Milvus.add_documents() 会把所有 metadata 字段都当成列来插入，
          但我们的自定义 Schema 里只有固定的 3 个用户字段（text / vector / source）。
          字段数量不匹配就会报 DataNotMatchException。

        解决方法：
          直接用 pymilvus MilvusClient 插入数据，只提供我们 Schema 里定义的字段。
          sparse_vector 字段由 Milvus 内部的 BM25 函数自动从 text 生成，无需我们手动填写。
        """
        from pymilvus import MilvusClient
        from chatchat.settings import Settings

        conn = Settings.kb_settings.kbs_config.get("milvus")
        uri = f"http://{conn.get('host', '127.0.0.1')}:{conn.get('port', '19530')}"
        client = MilvusClient(uri=uri)

        # 调用 Embedding 模型，批量把所有文档的文本转成向量
        embed_func = get_Embeddings(self.embed_model)
        texts = [doc.page_content for doc in docs]
        vectors = embed_func.embed_documents(texts)

        # 拼装成 Milvus 要求的插入格式：一个字典的列表
        # 每个字典代表一行数据，只需提供 Schema 中我们定义的字段
        # sparse_vector 字段由 Milvus BM25 函数自动生成，不需要我们填
        data = []
        for doc, vec in zip(docs, vectors):
            row = {
                "text": doc.page_content,
                "vector": vec,
                "source": str(doc.metadata.get("source", "")),
            }
            data.append(row)

        # 执行批量插入
        result = client.insert(collection_name=self.kb_name, data=data)

        # 把 Milvus 返回的 id 列表和原始 metadata 打包成上层需要的格式
        inserted_ids = result.get("ids", [])
        doc_infos = [
            {"id": str(id_), "metadata": doc.metadata}
            for id_, doc in zip(inserted_ids, docs)
        ]
        return doc_infos

    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        id_list = list_file_num_docs_id_by_kb_name_and_file_name(
            kb_file.kb_name, kb_file.filename
        )
        if self.milvus.col:
            self.milvus.col.delete(expr=f"pk in {id_list}")

        # Issue 2846, for windows
        # if self.milvus.col:
        #     file_path = kb_file.filepath.replace("\\", "\\\\")
        #     file_name = os.path.basename(file_path)
        #     id_list = [item.get("pk") for item in
        #                self.milvus.col.query(expr=f'source == "{file_name}"', output_fields=["pk"])]
        #     self.milvus.col.delete(expr=f'pk in {id_list}')

    def do_clear_vs(self):
        if self.milvus.col:
            self.do_drop_kb()
            self.do_init()


if __name__ == "__main__":
    # 测试建表使用
    from chatchat.server.db.base import Base, engine

    Base.metadata.create_all(bind=engine)
    milvusService = MilvusKBService("test")
    # milvusService.add_doc(KnowledgeFile("README.md", "test"))

    print(milvusService.get_doc_by_ids(["444022434274215486"]))
    # milvusService.delete_doc(KnowledgeFile("README.md", "test"))
    # milvusService.do_drop_kb()
    # print(milvusService.search_docs("如何启动api服务"))
