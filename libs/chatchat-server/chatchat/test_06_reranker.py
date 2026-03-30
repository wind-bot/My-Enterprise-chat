"""
测试 Reranker 精排效果
对比维度：
  精排前：hybrid_search 返回的原始顺序（基于向量距离+BM25）
  精排后：Cross-Encoder 重新打分后的顺序（基于语义相关性）
观察：两次返回的文档内容是否一样？顺序有没有变化？
"""

from chatchat.server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService
from chatchat.server.reranker.reranker import LangchainReranker
from chatchat.settings import Settings

KB_NAME = "samples"
QUERY = "如何解决连接 Milvus 失败的问题？"

def test_reranker_comparison():
    print("=" * 60)
    print("  Reranker 精排效果对比测试")
    print("=" * 60)

    kb_service = MilvusKBService(KB_NAME)
     # ── ① 精排前（粗召回 Top-5）────────────────────────────
    print(f"\n【精排前】hybrid_search 召回 Top-5（原始排序）")
    print(f"  排序依据：向量距离 + BM25 词频")
    print("-" * 60)
    # 临时关闭 Reranker 其实没用，系统严守 kb_settings.yaml
    # 所以我们直接调底层的 Retriever（纯 Hybrid）绕过 do_search 的保护
    from chatchat.server.file_rag.utils import get_Retriever
    kb_service._load_milvus()
    retriever = get_Retriever("milvushybrid").from_vectorstore(
        kb_service.milvus,
        top_k=15,  # 模拟粗召回 15 条
        score_threshold=0.0,
    )
    raw_docs = retriever.get_relevant_documents(QUERY)

    for i,doc in enumerate(raw_docs[:]):  # 只打印前5名看看
        print(f"【第{i+1}名】{doc.page_content[:100].replace(chr(10),'')}...")

     # ── ② 精排后（Cross-Encoder Top-3）─────────────────────
    print(f"\n【精排后】Reranker 重新打分，保留 Top-3")
    print(f"  排序依据：[Query + Doc] 送入 Cross-Encoder 的语义相关性分数")
    print("-" * 60)

    reranker = LangchainReranker(
        model_name_or_path=Settings.kb_settings.RERANKER_MODEL,
        top_n=3,
        device='cpu',
        max_length=Settings.kb_settings.RERANKER_MAX_LENGTH,        
    )

    reranker_docs = reranker.compress_documents(documents=raw_docs,query=QUERY)

    for i,doc in enumerate(reranker_docs):
        score = doc.metadata.get("relevance_score","N/A")
        score_str = f"{float(score):.4f}" if score is not None else "N/A"
        print(f"  [{i+1}] 精排分数={score_str}")
        print(f"       {doc.page_content[:100].replace(chr(10), ' ')}...")
    print("\n" + "=" * 60)
    print("  观察要点：")
    print("  - 精排后第 1 名，在精排前的排名是第几？")
    print("  - 有没有文档被从靠后位置提到前面？（这就是精排的价值！）")
    print("=" * 60)
if __name__ == "__main__":
    test_reranker_comparison()

    

    



    
    