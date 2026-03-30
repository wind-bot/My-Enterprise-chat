#测试 基于milvus的 向量搜索和BM25搜索
"""
测试链路：
  发送 query
    → MilvusKBService.do_search()   # 我们改造的入口
    → MilvusHybridRetrieverService  # 我们新写的检索器
    → Milvus hybrid_search API      # 数据库底层双路检索
    → 返回 Document 列表
"""
import traceback
from chatchat.server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService
from chatchat.settings import Settings
from chatchat.server.utils import get_Embeddings

# 替换成知识库中真实存在的名字（在 kb_settings.yaml 里配置的知识库名）
KB_NAME = "samples"

def test_hybrid_search():
    print("=" * 50)
    print("  测试：Milvus 原生混合检索")
    print("=" * 50)

    #初始化知识库服务
    kb_service = MilvusKBService(KB_NAME)
    # 准备测试问题
    query = "如何启动 API 服务？"
    print(f"\n查询问题:{query}")
    print("-" * 50)
    # 执行混合检索
    try:
        docs = kb_service.do_search(
            query = query,
            top_k=3,
            score_threshold=0.1
        )
        if not docs:
            print("未检索到相关文档,请确认知识库中已有数据")
        else:
            print(f"成功检索到{len(docs)}个文档：\n")
            for i,doc in enumerate(docs):
                print(f"----第{i+1}个文档----")
                print(f"来源：{doc.metadata.get('source','未知')}")
                print(f"内容：{doc.page_content[:150]   }...")
                print()
    except Exception as e:
        print(f"检索失败：{e}")
        import traceback
        traceback.print_exc()


def test_score_comparison():
    """
    分别执行三次搜索，对比三路结果的分数：
      ① Dense 向量搜索（语义相似度）  → 余弦相似度分数（越接近1越好）
      ② BM25 稀疏搜索（关键词匹配）   → BM25 分数（越大越相关）
      ③ RRF 混合搜索（两路融合排序）  → RRF 综合分数

    这样可以直观看到：有些文档在语义上相关但关键词不匹配，
    有些文档关键词命中但语义可能偏差，RRF 则取两者之长。
    """
    from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

    TOP_K = 5  # 每路搜索返回 5 条，方便对比

    conn = Settings.kb_settings.kbs_config.get("milvus")
    uri = f"http://{conn.get('host','127.0.0.1')}:{conn.get('port','19530')}"
    client = MilvusClient(uri=uri)

    embed_model = Settings.model_settings.DEFAULT_EMBEDDING_MODEL
    embed_func = get_Embeddings(embed_model)

    query = "如何启动 API 服务？"

    print("\n" + "=" * 60)
    print(f"  三路检索分数对比  |  查询：{query}")
    print("=" * 60)

    # ──────────────────────────────────────────
    # ① Dense 向量搜索（语义理解路）
    # ──────────────────────────────────────────
    print("\n【① Dense 向量搜索】  用 embedding 向量计算语义相似度")
    print("  分数含义：内积(IP)，越接近 1.0 说明语义越相似")
    print("-" * 60)

    query_vec = embed_func.embed_query(query)
    dense_results = client.search(
        collection_name=KB_NAME,
        data=[query_vec],
        anns_field="vector",
        search_params={"metric_type": "IP", "params": {"ef": 200}},
        limit=TOP_K,
        output_fields=["text", "source"],
    )[0]

    for i, hit in enumerate(dense_results):
        title = hit["entity"].get("text", "")[:60].replace("\n", " ")
        print(f"  [{i+1}] 分数={hit['distance']:.4f}  来源={hit['entity'].get('source','?')}")
        print(f"       内容片段: {title}...")

    # ──────────────────────────────────────────
    # ② BM25 稀疏搜索（关键词精确路）
    # ──────────────────────────────────────────
    print("\n【② BM25 关键词搜索】  直接匹配词频/逆文档频率")
    print("  分数含义：BM25 权重值，同一 query 下越大代表词频匹配越好")
    print("-" * 60)

    sparse_results = client.search(
        collection_name=KB_NAME,
        data=[query],                       # 传原始文本，Milvus 内部做 BM25 分词
        anns_field="sparse_vector",
        search_params={"metric_type": "BM25"},
        limit=TOP_K,
        output_fields=["text", "source"],
    )[0]

    for i, hit in enumerate(sparse_results):
        title = hit["entity"].get("text", "")[:60].replace("\n", " ")
        print(f"  [{i+1}] 分数={hit['distance']:.4f}  来源={hit['entity'].get('source','?')}")
        print(f"       内容片段: {title}...")

    # ──────────────────────────────────────────
    # ③ RRF 混合搜索（融合排名）
    # ──────────────────────────────────────────
    print("\n【③ RRF 混合搜索】  融合两路排名，取长补短")
    print("  分数含义：RRF(k=60) 综合分，两路排名都靠前则分数高")
    print("-" * 60)

    dense_req = AnnSearchRequest(
        data=[query_vec], anns_field="vector",
        param={"metric_type": "IP", "params": {"ef": 200}}, limit=TOP_K * 5,
    )
    sparse_req = AnnSearchRequest(
        data=[query], anns_field="sparse_vector",
        param={"metric_type": "BM25"}, limit=TOP_K * 5,
    )
    hybrid_results = client.hybrid_search(
        collection_name=KB_NAME,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=TOP_K,
        output_fields=["text", "source"],
    )[0]

    for i, hit in enumerate(hybrid_results):
        title = hit["entity"].get("text", "")[:60].replace("\n", " ")
        print(f"  [{i+1}] RRF分数={hit['distance']:.6f}  来源={hit['entity'].get('source','?')}")
        print(f"       内容片段: {title}...")

    print("\n" + "=" * 60)
    print("  观察要点：")
    print("  - 同一文档在三张表里排名是否一样？")
    print("  - 哪些文档在 Dense 高但 BM25 低（语义相关但关键词不命中）？")
    print("  - 哪些文档在 BM25 高但 Dense 低（关键词命中但语义偏差）？")
    print("  - RRF 排名靠前的，是否比单路结果更准？")
    print("=" * 60)


if __name__ == "__main__":
    test_hybrid_search()
    test_score_comparison()
