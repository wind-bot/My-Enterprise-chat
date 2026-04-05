"""
测试 Parent-Child 分层检索效果
测试维度：
  1. 文档入库时，如何被切分为父文档与子文档，并绑定 UUID。
  2. 检索时，纯向量/稀疏搜索命中的极细小片段（Child）。
  3. 劫持拉取：利用 Metadata 中的 parent_id 反向拉出完整原文段落（Parent）。
观察：检索出的小叶子内容，能否成功偷天换日变成完整的大树干！
"""
import os
import sys

# 确保能找到 chatchat 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(server_dir)

from chatchat.server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService

# 我们用你刚才手工重建并导入了持久化真实数据的库
KB_NAME = "samples"
# 选一个你之前在 test_06 里测试过的比较细节的查询词
QUERY = "那个数据库连不上了怎么整？"

def test_real_data_parent_child():
    print("=" * 60)
    print(f"  Parent-Child 真实库 ({KB_NAME}) 效果评估")
    print("=" * 60)

    print(f"\n【阶段一】加载现存知识库 {KB_NAME}")
    print("-" * 60)
    milvusService = MilvusKBService(KB_NAME)
    milvusService._load_milvus()
    print("  > 成功挂载持久化 Milvus 向量集合。新定义的 parent_id 表结构已就绪。")
    
    # 模拟极限长尾词检索
    print(f"\n【阶段二】底层偷窥：极细粒度词的长尾拦截（假装我们没有做父子召回）")
    print("-" * 60)
    print(f"  查询词 (query)：'{QUERY}'")
    
    # 手工抓取底层的 Retriever 看一眼
    from chatchat.server.file_rag.utils import get_Retriever
    retriever = get_Retriever("milvushybrid").from_vectorstore(
        milvusService.milvus,
        top_k=3,
        score_threshold=0.0,
        search_kwargs={"expr": "is_parent == false"} # 刻意只搜小碎块
    )
    
    raw_child_docs = retriever.get_relevant_documents(QUERY)
    
    if not raw_child_docs:
        print("  [空] 底层没有找到匹配到的小块记录，你可能需要换一个提问词。")
    else:
        print("\n  [偷窥] 如果不加我们的代码，传统框架默认捞出来的零碎切片(Child)如下：")
        for i, child in enumerate(raw_child_docs):
            print(f"  子片段 {i+1}，归属家谱 parent_id={child.metadata.get('parent_id')}")
            print(f"       文本片段: {child.page_content.replace(chr(10), ' ')}")

    
    # 拦截并组装完整结果
    print("\n【阶段三】偷天换日：我们的改造代码拦截拼装完整长文父节点（最终喂给 LLM 的部分）")
    print("-" * 60)
    # 调用我们的 do_search 主入口（它内部完成了 Child -> Parent 的拉取拼接）
    # 如果系统配置了 Reranker 它还会做精排，不过返回内容将已经是大段落了
    final_docs = milvusService.do_search(QUERY, top_k=2, score_threshold=0.0)
    
    if not final_docs:
        print("  [空] 没有任何搜索结果返回。")
    else:
        for i, d in enumerate(final_docs):
            print(f"  [最终返回结果 {i+1}]")
            print(f"       来源文件: {d.metadata.get('source')}")
            print(f"       家谱 ID : {d.metadata.get('parent_id')}")
            print(f"       身份标记 (is_parent): {d.metadata.get('is_parent')}")
            print(f"       组合还原后文本 (上下文完整):\n       => {d.page_content.replace(chr(10), ' ')}")

    print("\n" + "=" * 60)
    print("  【面试展示精华结论】：")
    print("  - 你可以对比一下【阶段二】和【阶段三】里的文本内容。")
    print("  - 阶段二暴露了 RAG 传统弊病：文本一刀两断，极其细碎。如果这丢给 LLM，极可能没主语、没逻辑前置条件。")
    print("  - 阶段三的内容则是我们在代码做了拦截替换，靠着相同 parent_id 从库里生拉硬拽回来的巨幅连续长文！")
    print("  - => 这就是完美兼顾了向量匹配高精准度与 LLM 推理上下文连贯性的硬核重构。")
    print("=" * 60)

if __name__ == "__main__":
    test_real_data_parent_child()
