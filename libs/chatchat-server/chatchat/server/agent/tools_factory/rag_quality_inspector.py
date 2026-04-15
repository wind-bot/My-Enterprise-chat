"""
RAG 检索质量检查工具
作用：对一个查询+知识库的组合，评估检索结果的质量
直接调用你改造过的 MilvusKBService.do_search() 混合检索管线
"""
from importlib.metadata import metadata
from pandas import describe_option
from email.policy import default
from chatchat.server.agent.tools_factory.tools_registry import regist_tool
from chatchat.server.knowledge_base.kb_doc_api import search_docs
from chatchat.server.pydantic_v1 import Field
from chatchat.server.utils import get_tool_config
from langchain_chatchat.agent_toolkits.all_tools.tool import BaseToolOutput 


def _evaluate_retrieval_quality(query:str,docs:list) -> dict:
    """
    评估检索结果质量
    返回包含质量指标的字典
    """
    if not docs:
        return {
            "quality_score":0,
            "quality_level":"极差",
            "reason":"未检索到任何文档",
            "suggestion":"请检查知识库是否有相关内容，或尝试换个关键词"
        }

    # 指标1:文档数量
    doc_count = len(docs)
    
    # 指标2:平均相关性分数(score 越低越相关，这是向量距离)
    scores = [doc.score for doc in docs if hasattr(doc,"score") and doc.score is not None]
    avg_score = sum(scores) / len(scores) if scores else None

    # 指标3:文档覆盖率(查询词在文档标题或内容中出现的比例)
    query_keywords = set(query.replace(",","").replace("."," ").split())
    keyword_hits = 0
    for doc in docs:
        content = doc.page_content if hasattr(doc,'page_content') else str(doc)
        if any(kw in content for kw in query_keywords if len(kw) > 1):
            keyword_hits += 1
    # 计算覆盖率
    keyword_coverage = keyword_hits / len(query_keywords) if query_keywords else 0

    # 综合评分
    if doc_count >= 3 and keyword_coverage >= 0.6:
        quality_level = "良好"
        quality_score = 85
    elif doc_count >=2 and keyword_coverage >= 0.4:
        quality_level = "一般"
        quality_score = 65
    elif doc_count >=1:
        quality_level = "较差"
        quality_score = 40
    else:
        quality_level = "极差"
        quality_score = 0
    
    return {
        "quality_score":quality_score,
        "quality_level":quality_level,
        "doc_count":doc_count,
        "keyword_coverage":f"{keyword_coverage:.0%}",
        "avg_relevance_score":f"{avg_score:.4f}" if avg_score else "N/A",
        "preview":[
            doc.page_content[:100] + "..."
            for doc in docs[:2]
            if hasattr(doc,'page_content')
        ]
    }

@regist_tool(title="RAG检索质检")
def rag_quality_inspector(
    query: str = Field(description="需要检索的问题或查询词"),
    kb_name: str = Field(description = "要检索的知识库名称"),
    top_k: int = Field(default=5,description ="检索文档数量，默认为5")
):
    """
    对RAG检索过程进行质量评估。
    当需要验证知识库能否回答某个问题、检查检索效果时使用。
    会返回检索到的文档数量、关键词覆盖率、相关性分数等质量指标。
    """
    try:
        "调用rag搜索"
        tool_config = get_tool_config("search_local_knowledgebase")
        docs = search_docs(
            query=query,
            knowledge_base_name=kb_name,
            top_k=top_k,
            score_threshold=tool_config.get("score_threshold",1.0),
            file_name="",
            metadata={},
        )

        #评估质量
        quality = _evaluate_retrieval_quality(query,docs)
        # 格式化输出
        result = (
            f"=== RAG 检索质量报告 ===\n"
            f"查询: {query}\n"
            f"知识库: {kb_name}\n"
            f"质量评级: {quality['quality_level']} (得分: {quality['quality_score']}/100)\n"
            f"检索文档数: {quality.get('doc_count', 0)}\n"
            f"关键词覆盖率: {quality.get('keyword_coverage', 'N/A')}\n"
            f"平均相关性分数: {quality.get('avg_relevance_score', 'N/A')}\n"
        )
        if quality.get("preview"):
            result += "\n前2条检索结果预览:\n"
            for i, preview in enumerate(quality["preview"], 1):
                result += f"{i}. {preview}\n"
        if "suggestion" in quality:
            result += f"\n建议: {quality['suggestion']}\n"
    except Exception as e:
        result = f"RAG质检过程出错: {str(e)}"
    return BaseToolOutput({"result": result})
        
