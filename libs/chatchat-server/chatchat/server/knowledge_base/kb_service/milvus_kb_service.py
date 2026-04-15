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
from chatchat.utils import build_logger

logger = build_logger()

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
            collection_name="enterprise_global_kb", #全局表
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
        # 连接数据库获取数据库对象
        conn = Settings.kb_settings.kbs_config.get("milvus")
        uri = f"http://{conn.get('host', '127.0.0.1')}:{conn.get('port', '19530')}"
        client = MilvusClient(uri=uri)

        #【核心改造点 1：把局部表提升为公司级大表】- 多用户分表
        # 把原本的 col_name = self.kb_name 删掉，改为全局固定表：
        col_name = "enterprise_global_kb"
        # 然后把传进来的 kb_name 降级作为物理分区名：
        partition_name = self.kb_name


        # 检查 collection 是否已经存在，且字段是否完整
        if client.has_collection(col_name):
            schema = client.describe_collection(col_name)
            field_names = [f["name"] for f in schema.get("fields", [])]
            if "sparse_vector" in field_names:
                # 已经有稀疏向量字段，先确保分区存在再返回
                if not client.has_partition(collection_name=col_name,partition_name=partition_name):
                    client.create_partition(collection_name=col_name,partition_name=partition_name)
                    print(f"[架构进化] 已在总库下，成功为租户 '{partition_name}' 开辟安全沙箱分区！")
                client.load_partitions(collection_name = col_name,partition_names = [partition_name])
                return
            # 字段不完整，删掉旧的
            client.drop_collection(col_name)

        # 关键：动态探测 Embedding 模型的真实输出维度，不能写死
        # 不同模型维度不一样：bge-m3 = 1024, embedding-3 = 2048, text-embedding-3-large = 3072
        embed_func = get_Embeddings(self.embed_model)
        test_vec = embed_func.embed_query("维度探测")
        actual_dim = len(test_vec)
        print(f"[MilvusKBService] 探测到 Embedding 维度：{actual_dim}")

        # 核心：用 pymilvus 原生 API 手动建 Schema，加入两种向量字段
        schema = client.create_schema()
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True) # 主键id
        schema.add_field("text", DataType.VARCHAR, max_length=65535,
                         enable_analyzer=True)                       # 开启分词分析器，供 BM25 使用 文本内容
        schema.add_field("vector", DataType.FLOAT_VECTOR,
                         dim=actual_dim)                             # Dense 向量（动态维度）
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)  # BM25 稀疏向量
        schema.add_field("source", DataType.VARCHAR, max_length=512,
                         default_value="")                           # 文件来源
        #=====父子分层检索新增字段========
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64, default_value="") # 父文档的家族ID
        schema.add_field("is_parent", DataType.BOOL, default_value=False)                # 标记身份：True为父，False为子

        # 定义 BM25 内置函数：自动把 text 字段的文本转换成 sparse_vector 稀疏向量
        # 把所有的text做分词处理，生成稀疏向量，这里没有建立索引
        bm25_function = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"], # 从text读取原始文本
            output_field_names=["sparse_vector"], # 自动生成，写入sparse_vector字段
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

        #【核心改造点 2：为每个库/部门专门创建和加载物理隔离的分区】
        if not client.has_partition(collection_name=col_name,partition_name=partition_name):
            client.create_partition(collection_name=col_name,partition_name=partition_name)
            print(f"[架构进化] 已在总库下，成功为租户 '{partition_name}' 开辟安全沙箱分区！")
        #为了进一步省内存，我们可以只加载（Load）特定的分区！
        client.load_partitions(collection_name = col_name,partition_names = [partition_name])


    def do_init(self):
        self._ensure_hybrid_collection()   # ← 新增：确保双字段 Collection 存在
        self._load_milvus()

    def do_drop_kb(self):
        # if self.milvus.col:
        #     self.milvus.col.release()
        #     self.milvus.col.drop()
        from pymilvus import MilvusClient
        from chatchat.settings import Settings
        # 1. 建立连接
        conn = Settings.kb_settings.kbs_config.get("milvus")
        uri = f"http://{conn.get('host','127.0.0.1')}:{conn.get('port','19530')}"
        client = MilvusClient(uri=uri)
        # 2. 目标表名
        col_name = "enterprise_global_kb"
        # 3. 目标分区名（即知识库名）
        partition_name = self.kb_name
        # 4. 检查分区是否存在，不存在则创建
        if client.has_partition(collection_name=col_name, partition_name=partition_name):
            #先卸载本分区的内存
            client.release_partitions(collection_name = col_name,partition_names=[partition_name])
            #再从物理硬盘上销毁本分区（绝对不伤害别的分区内容）
            client.drop_collection(collection_name=col_name,partition_names=[partition_name])
            print(f"[数据安全] 已安全销毁租户沙箱分区 '{partition_name}'，其他部门数据未受影响。")


    def do_search(self, query: str, top_k: int, score_threshold: float):
        """
        改造前：只走 Dense 向量检索
        改造后：走 Dense + BM25 Sparse 混合检索
            为什么要在这里改？
            因为 do_search 是知识库被调用时的入口函数，
            用户问一个问题 → 调用 do_search → 返回相关文档 → 这些文档被拼到 Prompt 里给 LLM
            只要改这一个函数，整个知识库的检索链路就全部升级了
        ----------
        改造二阶段召回+精排序 
        两阶段检索：粗召回(hybrid) + 精排(reranker)
        链路：
            1. hybrid_search 召回 Top-20 候选（多召回，宁多勿少）
            2. 如果 USE_RERANKER=true，用 Cross-Encoder 精排，取 Top-3
            3. 返回高质量文档给 LLM
        ---------
        改造三：增强查询
        原理：用 LLM 生成一段假设性的"专业解答"，
        把它的向量作为检索锚点，弥补口语化 query 的语义特征缺陷。
        为什么放在这里（do_search 最顶端）？
        因为要在进入任何检索引擎【之前】就把 query 变形，
        后面混合检索、父子回拉、Reranker 全都用增强后的 query，
        一次改动，链路全部升级。
        """
        from chatchat.server.utils import get_ChatOpenAI,get_default_llm
        from chatchat.settings import Settings

        try:
        # 思考二：temperature=0.7 是什么意思？为什么不用 0.0？
        # temperature 控制模型的"创造力"：
        # temperature=0.0 → 输出最保守、最确定的答案（适合精确问答）
        # temperature=0.7 → 输出有一定发散性（适合我们！要它多写术语，不要太死板）
        # temperature=1.0+ → 太随机，可能偏题
        # 这里选 0.7 是最佳平衡点。
            llm = get_ChatOpenAI(
                model_name=get_default_llm(),
                temperature=0.7,
                streaming=False,
            )
             # 思考四：这个 Prompt 是怎么设计的？为什么这样写？
            # 关键点1："不需要绝对正确" → 给 LLM 减压，让它大胆说术语，不要因为不确定就说废话
            # 关键点2："尽可能使用行业术语" → 直接命题，告诉 LLM 我们要什么
            # 关键点3："直接输出，不用寒暄" → 避免 LLM 输出"好的，以下是..." 这种废话前缀
            hyde_prompt = (
                f"你是一位资深的技术专家，请针对以下用户问题，写一段简明专业的解答。\n"
                f"要求：\n"
                f"1. 不需要绝对正确，但必须使用大量该领域的专业术语和缩写\n"
                f"2. 长度在 100-150 字之间\n"
                f"3. 直接输出解答内容，不用任何开场白\n"
                f"用户问题：{query}\n"
            )
            logger.info(f"[HyDE] 正在对用户 query 进行语义扩写增强...")
            # 调用LLM 拿到假设性答案
            response = llm.invoke(hyde_prompt)
            hypothetical_answer = response.content.strip()
            logger.info(f"[HyDE] 扩写完成，假设性答案：{hypothetical_answer[:80]}...")
            # 思考五：为什么要把原始 query 和假设性答案拼接？而不只用假设性答案？
            # 答：LLM 虽然会用正确术语，但偶尔会偏题（幻觉严重时）。
            # 拼接原始 query 起到"锚点"作用，防止语义完全漂移。
            # 这是工程实践的安全保障。
            enhanced_query = f"{query}\n{hypothetical_answer}"
        except Exception as e:
            #如果LLM超时就用原query
            logger.warning(f"[HyDE] LLM 超时或报错，将使用原始 query 进行检索：{e}")
            enhanced_query = query





        self._load_milvus()
        # embed_func = get_Embeddings(self.embed_model)
        # embeddings = embed_func.embed_query(query)
        # docs = self.milvus.similarity_search_with_score_by_vector(embeddings, top_k)
        #步骤一，粗排 混合检索
        retriever = get_Retriever("milvushybrid").from_vectorstore(
            self.milvus,
            top_k=top_k * 5,
            score_threshold=score_threshold,
            # 因为加了父子模块，只对比子段（注意：Milvus 的 sql 表达式中布尔值要用小写 false）
            search_kwargs = {
                "expr":"is_parent == false",
                "partition_names":[self.kb_name]
                }
        )
        docs = retriever.get_relevant_documents(enhanced_query)
        # 父子模块--此时的 docs 全是十几到几十个字的精准词条，我们把他们的家谱 ID 抽出来去重
        parent_ids = list(set([str(doc.metadata.get("parent_id")) for doc in docs if doc.metadata.get("parent_id")]))
        if parent_ids:
            from pymilvus import MilvusClient
            conn = Settings.kb_settings.kbs_config.get("milvus")
            uri = f"http://{conn.get('host', '127.0.0.1')}:{conn.get('port', '19530')}"
            client = MilvusClient(uri=uri)
            # 去 Milvus 中原生拉取这些 ID 对应的父文档（is_parent == true）
            id_str = "[" + ",".join([f"'{id}'" for id in parent_ids ]) + "]"
            parent_res = client.query(
                collection_name="enterprise_global_kb",
                partition_names = [self.kb_name], #这里去老祖宗库里捞数据时，也必须守规矩
                filter=f"parent_id in {id_str} and is_parent == true",
                output_fields = ["text", "source", "parent_id"]
            )
            # 整理成 Langchain Document 格式
            docs = []
            for p in parent_res:
                docs.append(Document(
                    page_content = p["text"],
                    metadata={"source": p["source"], "parent_id": p["parent_id"], "is_parent": True}
                ))
            logger.info(f"[Parent-Child] 小块命中，成功拉回 {len(parent_res)} 个宽泛语境大块！")
        # 步骤二、精排reranker
        if Settings.kb_settings.USE_RERANKER and len(docs) > 0:
            from chatchat.server.reranker.reranker import LangchainReranker
            reranker = LangchainReranker(
                model_name_or_path=Settings.kb_settings.RERANKER_MODEL,
                top_n = top_k,  # 精排后只留 top_k 条
                device="cpu",   # Mac 没有 CUDA，用 cpu
                max_length=Settings.kb_settings.RERANKER_MAX_LENGTH,
                )
            docs = reranker.compress_documents(documents=docs,query=query)
            logger.info(f"[Reranker] 精排完成，从 {top_k * 5} 条候选 → 保留 {len(docs)} 条")
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
        # texts = [doc.page_content for doc in docs]
        # vectors = embed_func.embed_documents(texts)

        # 拼装成 Milvus 要求的插入格式：一个字典的列表
        # 每个字典代表一行数据，只需提供 Schema 中我们定义的字段
        # sparse_vector 字段由 Milvus BM25 函数自动生成，不需要我们填
        # data = []
        # for doc, vec in zip(docs, vectors):
        #     row = {
        #         "text": doc.page_content,
        #         "vector": vec,
        #         "source": str(doc.metadata.get("source", "")),
        #     }
        #     data.append(row)

        #=============修改父子模块============
        # 调用 Chatchat 开源项目本土化定制的中文切分器（比官方 langchain 更懂中文段落）
        from chatchat.server.file_rag.text_splitter.chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
        import uuid

        # 定义一个 150 字的小刀，把长文档（Parent）按照中文语境切碎
        child_splitter = ChineseRecursiveTextSplitter(chunk_size=150, chunk_overlap=20)
        data = []
        all_text = []
        temp_records = []
        for doc in docs:
            # 1.父id
            parent_id = uuid.uuid4().hex
            # 2.存入完整的父节点并打上is_parent:True
            temp_records.append({
                "text":doc.page_content,
                "source":str(doc.metadata.get("source","")),
                "parent_id":parent_id,
                "is_parent":True,
            })
            all_text.append(doc.page_content)

            # 3.将大段落劈成小段落，也就是子段落,打上 is_parent: False，绑定同一个 parent_id
            child_docs = child_splitter.split_documents([doc])
            for child in child_docs:
                temp_records.append({
                    "text":child.page_content,
                    "source":str(child.metadata.get("source","")),
                    "parent_id":parent_id,
                    "is_parent":False
                })
                all_text.append(child.page_content)
        # 4.批量向量化-将大段和子段都向量化
        vectors = embed_func.embed_documents(all_text)
        # 5.将向量和数据拼装成milvus需要的格式
        for record,vec in zip(temp_records,vectors):
            record["vector"] = vec
            data.append(record)
        #=============修改父子模块============
        # 执行批量插入
        # 这里milvus 会计算BM25需要的向量表

        batch_size = 500
        inserted_ids = []

        for i in range(0, len(data), batch_size):
            batch_data = data[i : i + batch_size]
            result = client.insert(
                collection_name="enterprise_global_kb", 
                partition_name = self.kb_name,
                data=batch_data,
            )
            inserted_ids.extend(result.get("ids", []))

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
