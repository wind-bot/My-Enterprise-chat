from __future__ import annotations

import asyncio, json
import uuid
from typing import AsyncIterable, List, Optional, Literal

from fastapi import Body, Request
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate

from langchain_community.retrievers import BM25Retriever
from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory


from chatchat.settings import Settings
from chatchat.server.agent.tools_factory.search_internet import search_engine
from chatchat.server.api_server.api_schemas import OpenAIChatOutput
from chatchat.server.chat.utils import History
from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory
from chatchat.server.knowledge_base.kb_doc_api import search_docs, search_temp_docs
from chatchat.server.knowledge_base.utils import format_reference
from chatchat.server.utils import (wrap_done, get_ChatOpenAI, get_default_llm,
                                   BaseResponse, get_prompt_template, build_logger,
                                   check_embed_model, api_address
                                )


logger = build_logger()


async def kb_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                mode: Literal["local_kb", "temp_kb", "search_engine"] = Body("local_kb", description="知识来源"),
                kb_name: str = Body("", description="mode=local_kb时为知识库名称；temp_kb时为临时知识库ID，search_engine时为搜索引擎名称", examples=["samples"]),
                top_k: int = Body(Settings.kb_settings.VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                score_threshold: float = Body(
                    Settings.kb_settings.SCORE_THRESHOLD,
                    description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                    ge=0,
                    le=2,
                ),
                history: List[History] = Body(
                    [],
                    description="历史对话",
                    examples=[[
                        {"role": "user",
                        "content": "我们来玩成语接龙，我先来，生龙活虎"},
                        {"role": "assistant",
                        "content": "虎头虎脑"}]]
                ),
                stream: bool = Body(True, description="流式输出"),
                model: str = Body(get_default_llm(), description="LLM 模型名称。"),
                temperature: float = Body(Settings.model_settings.TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
                max_tokens: Optional[int] = Body(
                    Settings.model_settings.MAX_TOKENS,
                    description="限制LLM生成Token数量，默认None代表模型最大值"
                ),
                prompt_name: str = Body(
                    "default",
                    description="使用的prompt模板名称(在prompt_settings.yaml中配置)"
                ),
                return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM"),
                request: Request = None,
                ):
    
    if mode == "local_kb":
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb is None:
            return BaseResponse(code=404, msg=f"未找到知识库 {kb_name}")
    #异步迭代器--用来处理流式输出
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            nonlocal history, prompt_name, max_tokens
          
            history = [History.from_data(h) for h in history]
            # 1 根据用户的query 和 mode 来决定用什么方式去召回文档--也就是查向量数据库获取跟用户提问相关的文档。
            if mode == "local_kb":
                kb = KBServiceFactory.get_service_by_name(kb_name) #知识库实例
                ok, msg = kb.check_embed_model() #检查向量模型是否可用
                if not ok:
                    raise ValueError(msg)
                #核心 根据用户问题去召回文档--也就是查向量数据库获取跟用户提问相关的文档。
                #向量检索
                docs = await run_in_threadpool(search_docs, 
                                                query=query, #用户查询
                                                knowledge_base_name=kb_name, #知识库名称
                                                top_k=top_k, #召回文档数量
                                                score_threshold=score_threshold, #召回文档相关度阈值
                                                file_name="",
                                                metadata={})
                

                # #bm25稀疏检索
                # bm25_docs = my_bm25_search(query, kb_name, top_k)
                # docs = deduplicate_and_rrf(vector_docs,bm25_docs,top_k=top_k)

                 #整理引用来源格式 (会在前端展示"出处：[1] xxx.pdf")
                source_documents = format_reference(kb_name, docs, api_address(is_public=True))

                """
                这是学习bm25的思路和过程
                #bm25稀疏检索--要实现 BM25 关键词检索，我们需要用到你计划里说的 `rank_bm25` 库。想要进行词频匹配，我们需要把当前知识库的所有 chunk 都调出来给它做对比。
                bm25_docs = my_bm25_search(query, kb_name, top_k)
                #合并向量检索和 bm25_doc稀疏检索的结果
                merged_docs = vector_docs+bm25_docs
                #对合并后的结果进行去重
                final_docs = deduplicate(merged_docs)
                #把筛选后的结果给大模型
                docs = final_docs[:top_k]
                """
                #bm25
            elif mode == "temp_kb":
                ok, msg = check_embed_model()
                if not ok:
                    raise ValueError(msg)
                docs = await run_in_threadpool(search_temp_docs,
                                                kb_name,
                                                query=query,
                                                top_k=top_k,
                                                score_threshold=score_threshold)
                source_documents = format_reference(kb_name, docs, api_address(is_public=True))
            elif mode == "search_engine":
                result = await run_in_threadpool(search_engine, query, top_k, kb_name)
                docs = [x.dict() for x in result.get("docs", [])]
                source_documents = [f"""出处 [{i + 1}] [{d['metadata']['filename']}]({d['metadata']['source']}) \n\n{d['page_content']}\n\n""" for i,d in enumerate(docs)]
            else:
                docs = []
                source_documents = []
            # import rich
            # rich.print(dict(
            #     mode=mode,
            #     query=query,
            #     knowledge_base_name=kb_name,
            #     top_k=top_k,
            #     score_threshold=score_threshold,
            # ))
            # rich.print(docs)
            if return_direct:
                yield OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    model=None,
                    object="chat.completion",
                    content="",
                    role="assistant",
                    finish_reason="stop",
                    docs=source_documents,
                ) .model_dump_json()
                return
            #是用来拦截 LLM 吐出的每个字串
            callback = AsyncIteratorCallbackHandler()
            callbacks = [callback]

            # Enable langchain-chatchat to support langfuse -- 支持可视化追查
            import os
            langfuse_secret_key = os.environ.get('LANGFUSE_SECRET_KEY')
            langfuse_public_key = os.environ.get('LANGFUSE_PUBLIC_KEY')
            langfuse_host = os.environ.get('LANGFUSE_HOST')
            if langfuse_secret_key and langfuse_public_key and langfuse_host :
                from langfuse import Langfuse
                from langfuse.callback import CallbackHandler
                langfuse_handler = CallbackHandler()
                callbacks.append(langfuse_handler)

            if max_tokens in [None, 0]:
                max_tokens = Settings.model_settings.MAX_TOKENS
            # 实例化大模型
            llm = get_ChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbacks,
            )
            # TODO： 视情况使用 API
            # # 加入reranker
            # if Settings.kb_settings.USE_RERANKER:
            #     reranker_model_path = get_model_path(Settings.kb_settings.RERANKER_MODEL)
            #     reranker_model = LangchainReranker(top_n=top_k,
            #                                     device=embedding_device(),
            #                                     max_length=Settings.kb_settings.RERANKER_MAX_LENGTH,
            #                                     model_name_or_path=reranker_model_path
            #                                     )
            #     print("-------------before rerank-----------------")
            #     print(docs)
            #     docs = reranker_model.compress_documents(documents=docs,
            #                                              query=query)
            #     print("------------after rerank------------------")
            #     print(docs)
            context = "\n\n".join([doc["page_content"] for doc in docs])
            #2.获取大模型提示词模板，并把查询到的文档和用户query 一起拼接到prompt模版中
            if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
                prompt_name = "empty"
                # 获取提示词模板并把 搜索知识库的结果拼接到提示词中
            prompt_template = get_prompt_template("rag", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])

            #3.后台开启一个异步任务，LLM根据   context内容 和 用户原始提问，启动链，继续获得回答！
            #LangChain 的 LCEL 语法（也就是 | 管道符）把刚才做好的“待填空大模板” 和 LLM 绑在了一根水管上
            chain = chat_prompt | llm

            # Begin a task that runs in the background.- 把刚才在向量库查到的多个docs 拼合的长文本塞进第一个孔，# 把用户前端输入的真实的 query 塞进第二个孔
            task = asyncio.create_task(wrap_done(
                chain.ainvoke({"context": context, "question": query}),
                callback.done),
            )

            if len(source_documents) == 0:  # 没有找到相关文档
                source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

            if stream:
                # yield documents first
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion.chunk",
                    content="",
                    role="assistant",
                    model=model,
                    docs=source_documents,
                )
                yield ret.model_dump_json()

                async for token in callback.aiter():
                    ret = OpenAIChatOutput(
                        id=f"chat{uuid.uuid4()}",
                        object="chat.completion.chunk",
                        content=token,
                        role="assistant",
                        model=model,
                    )
                    yield ret.model_dump_json()
            else:
                answer = ""
                async for token in callback.aiter():
                    answer += token
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion",
                    content=answer,
                    role="assistant",
                    model=model,
                )
                yield ret.model_dump_json()
            await task
        except asyncio.exceptions.CancelledError:
            logger.warning("streaming progress has been interrupted by user.")
            return
        except Exception as e:
            logger.error(f"error in knowledge chat: {e}")
            yield {"data": json.dumps({"error": str(e)})}
            return

    if stream:
        return EventSourceResponse(knowledge_base_chat_iterator())
    else:
        return await knowledge_base_chat_iterator().__anext__()

#混合检索后的docs去重
def deduplicate_and_rrf(vector_docs:List[Document],bm25_docs:List[Document] ,top_k=5,rrf_k=60):
    """
    RRF(分数对齐)混合排序 + 去重
    倒数排名融合 -最终得分 = 1 / (60+在A中的最终排名) + 1/(60+在B中的排名)
    """
    rrf_scores = {} #记录每篇文档的最终得分
    #利用python字典的数据结构key不能重复的特点，来去重，--hash去重
    unique_docs = {} #存放去掉重复后的真实 Document 对象
    #1. 给向量召回的文档算 RRF分数
    for rank,doc in enumerate(vector_docs):
        fingerprint = doc.get("page_content","")
        if fingerprint not in unique_docs:
            unique_docs[fingerprint] = doc
            rrf_scores[fingerprint] = 0.0
        
        rrf_scores[fingerprint] += 1 / (rrf_k+rank + 1)
    #2.给 BM25 召回的文档算RRF 分数
    for rank,doc in enumerate(bm25_docs):
        fingerprint = doc.get("page_content","")
        if fingerprint not in unique_docs:
            unique_docs[fingerprint] = doc
            rrf_scores[fingerprint] = 0.0
        
        rrf_scores[fingerprint] += 1 / (rrf_k+rank + 1)
    
    #3.把字典变为列表，并根据刚才算出来的真实RRF分数进行降序排序
    sorted_fingerprints = sorted(rrf_scores.keys(),key = lambda x: rrf_scores[x],reverse=True)

    #返回top_k个文档
    final_docs = [unique_docs[fp] for fp in sorted_fingerprints[:top_k]]
    return final_docs
    

def my_bm25_search(query:str,kb_name:str,top_k:int=5):
    """
    轻量级内存版BM25 稀疏检索函数
    """
    # 1.拿到当前用户的知识库服务层 - KB Service
    kb = KBServiceFactory.get_service_by_name(kb_name)
    if kb is None:
        return[]

    # 2.核心操作，强行打开底层向量数据库的底座（比如FAISS），直接把向量数据库中所有切分好的原始文本（Chunk）抽出来。
    with kb.load_vector_store().acquire() as vs:
        # vs.docstore._dict 里存放着我们在 Day2/Day3 辛辛苦苦存进去的所有带 metadata 的 Document 对象
        all_docs = list(vs.docstore._dict.values())
    if not all_docs:
        return [] 
    #3. 用langchain封装好的BM25Retriever 它底层调用的就是 rank_bm25库，内置了中文支持机制
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    #4.指定要topk
    bm25_retriever.k = top_k
    #5.执行检索 - 根据词频算法，在全局帮我把包含关键词的文档找出来
    results = bm25_retriever.invoke(query)
    #强制将原生 Document 对象序列化转成字典格式，和原版向量接口保持绝对一致！
    dict_results = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "score": 0.0
        }
        for doc in results
    ]
    return dict_results
