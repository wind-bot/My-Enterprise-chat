"""
知识库状态查询工具
作用：让 Agent 在回答问题前先了解有哪些知识库可用，以及各库的基本信息
"""
from chatchat.server.agent.tools_factory.tools_registry import regist_tool
from chatchat.server.knowledge_base.kb_api import list_kbs
from chatchat.server.knowledge_base.kb_doc_api import list_files
from chatchat.server.pydantic_v1 import Field
from langchain_chatchat.agent_toolkits.all_tools.tool import BaseToolOutput


@regist_tool(title="知识库状态查询")
def kb_retrieval_stats(
    kb_name: str = Field(
        default="",
        description="知识库名称，留空则列出所有知识库的概况"
    ),
):
    """
    查询知识库的基本信息和状态。
    当需要了解"有哪些可以查询的知识库"、"某个知识库有多少文档"时使用此工具。
    留空 kb_name 获取所有知识库列表；填写 kb_name 获取该库的详细文档列表。
    """
    try:
        if not kb_name:
            # 列出所有知识库
            kbs = list_kbs().data
            if not kbs:
                result = "当前没有任何知识库。"
            else:
                lines = ["当前可用知识库列表：\n"]
                for kb in kbs:
                    # 每个 kb 对象有 kb_name, kb_info, vs_type 等字段
                    lines.append(
                        f"- 知识库名称: {kb.kb_name}\n"
                        f"  简介: {kb.kb_info or '暂无简介'}\n"
                        f"  向量库类型: {kb.vs_type}\n"
                    )
                result = "\n".join(lines)
        else:
            # 查询指定知识库的文档列表
            docs_response = list_files(knowledge_base_name=kb_name)
            if docs_response.code != 200:
                result = f"知识库 '{kb_name}' 不存在或查询失败。"
            else:
                docs = docs_response.data
                if not docs:
                    result = f"知识库 '{kb_name}' 中暂无文档。"
                else:
                    lines = [f"知识库 '{kb_name}' 共有 {len(docs)} 个文档：\n"]
                    for doc in docs[:10]:  # 最多显示 10 个
                        lines.append(
                            f"- {doc.get('file_name', '未知文件')}"
                            f"  (大小: {doc.get('file_size', 0)} bytes)"
                        )
                    if len(docs) > 10:
                        lines.append(f"... 还有 {len(docs) - 10} 个文档未显示")
                    result = "\n".join(lines)

    except Exception as e:
        result = f"查询知识库状态时出错：{str(e)}"

    return BaseToolOutput({"result": result})
