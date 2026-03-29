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
    print(f"\n查询问题{query}")
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

if __name__ == "__main__":
    test_hybrid_search()
            




