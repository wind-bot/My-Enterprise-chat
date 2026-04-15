"""
智能知识库路由工具
作用：根据用户问题，自动选择最合适的知识库，然后进行检索
解决了"用户不知道该查哪个库"的工程问题
实现：用 Embedding 计算查询和知识库简介的语义相似度
"""

import numpy as np
from chatchat.server.agent.tools_factory.tools_registry import regist_tool
from chatchat.server.knowledge_base.kb_api import list_kbs
from chatchat.server.knowledge_base.kb_doc_api import search_docs
from chatchat.server.pydantic_v1 import Field
from chatchat.server.utils import get_Embeddings, get_tool_config
from langchain_chatchat.agent_toolkits.all_tools.tool import BaseToolOutput
from chatchat.server.agent.tools_factory.tools_registry import format_context


def _consine_similarity(vect1:list,vect2:list) -> float:
    #计算两个向量的余弦相似度
    a = np.array(vect1)
    b = np.array(vect2)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b)))


def _select_best_kb(query:str , kbs: list) -> tuple:
    """
    用embedding 相似度选出最匹配的知识库
    返回(最佳知识库名，相似度得分，候选排名列表)
    """
    #获取embedding模型 与rag同一个
    embeddings = get_Embeddings()
    #对查询做embedding
    query_vec = embeddings.embed_query(query)

    #对每个知识库的简介做embedding，计算相似度
    kb_scores = []
    for kb in kbs:
        # kb_info 是用户建库时填的简介，这就是为什么建库时 kb_info 很重要！
        kb_decs = kb.kb_info or kb.kb_name
        kb_vec = embeddings.embed_query(kb_decs)
        sim = _consine_similarity(query_vec,kb_vec)
        kb_scores.append({
            "kb_name":kb.kb_name,
            "kb_info":kb.kb_info,
            "similarity":sim
        })
    
    #按相似度排名
    kb_scores.sort(key = lambda x : x["similarity"],reverse=True)
    best = kb_scores[0]
    return best["kb_name"],best["similarity"],kb_scores

@regist_tool(title="智能知识库路由检索")
def smart_kb_router(
    query:str = Field(description = "用户的问题或查询内容，请务必使用中文进行思考和回复"),
    top_k:int = Field(destription = "从最佳知识库中检索的文档数量")
):
    """
    智能路由检索工具：自动根据问题语义，选择最合适的知识库进行检索。
    当用户问题不明确属于哪个知识库，或希望系统自动选择时使用。
    内部通过 Embedding 相似度匹配查询和知识库简介，选出最相关的库进行检索。
    """
    try:
        #第一步 获取所有可用知识库
        kbs = list_kbs().data
        if not kbs:
            return BaseToolOutput({"result":"当前没有任何可用的知识库"})
        if len(kbs) == 1:
            #只有一个库，直接用
            best_kb = kbs[0].kb_name
            routing_info = f"仅有一个知识库'{best_kb},直接使用'\n"
        else:
            #第二步 用Embedding计算最佳知识库
            best_kb,best_score,all_scores = _select_best_kb(query,kbs)
            routing_info = (
                f"智能路由结果:选择知识库'{best_kb}'"
                f"相似度:'{best_score:.3f}'"
                f"候选排名：" +
                " > ".join([f"{s['kb_name']}({s['similarity']:.3f})" for s in all_scores[:3]])
                + "\n\n"
            )
        #第三步 在最佳知识库中检索
        tool_config = get_tool_config("search_local_knowledgebase")
        docs = search_docs(
            query = query,
            knowledge_base_name=best_kb,
            top_k=top_k,
            score_threshold=tool_config.get("score_threshold",1.0),
            file_name="",
             metadata={},
        )

        if not docs:
            result = routing_info + f"在知识库'{best_kb}'中未找到相关内容"
        else:
            #格式化检索结果（和 search_local_knowledgebase 一样的格式）
            context = ""
            for doc in docs:
                content = doc.get("page_content", "") if isinstance(doc, dict) else getattr(doc, "page_content", "")
                context += content + "\n\n"
            result = routing_info + f"从知识库'{best_kb}'检索到以下内容\n\n" + context
    except Exception as e:
        result = f"智能路由检索出错：{str(e)}"
    return BaseToolOutput({"result":result})






