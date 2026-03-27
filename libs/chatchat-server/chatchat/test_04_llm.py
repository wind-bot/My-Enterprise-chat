import os
import sys

# 1. 挂载环境
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
os.environ["CHATCHAT_ROOT"] = project_root

# ====================================================================
# 第一步：连接到底层的 FAISS 向量库，动态召回知识碎片（彻底告别硬编码！）
# ====================================================================
from chatchat.server.knowledge_base.kb_service.faiss_kb_service import FaissKBService

kb_name = "test_faiss_kb"
print(f"[INFO] 1. 正在苏醒底层的 FAISS 物理向量库：{kb_name}...")
faiss_service = FaissKBService(kb_name, embed_model="embedding-3")

# 因为小抄里没有“北京”这两个字，所以我们不能问它叫什么名字！我们只要问大模型只凭这句话能提炼出什么其他客观事实。
user_query = "那个有很多刚毕业年轻人挥洒汗水的地方，有哪些类型的机构设立了总部？"
print(f"❓ [用户提问]: {user_query}")

print("[INFO] FAISS 启动！正在根据你的提问，在三维数字空间里寻找最匹配的碎片...")
with faiss_service.load_vector_store().acquire() as vs:
    search_results_with_scores = vs.similarity_search_with_score(user_query, k=2)

print("\n✅ FAISS 捞回了以下真实存在的底层数据：")
# ★ 最关键的衔接点：把捞出来的多个离散片段（page_content），用换行符强行拼接到一起，形成最终小抄！
retrieved_context_list = []
for i, (doc, score) in enumerate(search_results_with_scores):
    print(f"   [片段 {i+1}] (距离得分:{score:.4f}): {doc.page_content[:30]}...")
    retrieved_context_list.append(doc.page_content)

# 用两条换行符把多块碎片粘成一段连贯的大文章
retrieved_context = "\n\n".join(retrieved_context_list)


# ====================================================================
# 第二步：去系统的 prompt_settings.yaml 里把法术咒语（Prompt模板）拿出来
# ====================================================================
from chatchat.server.utils import get_prompt_template
from langchain.prompts import PromptTemplate

raw_template = get_prompt_template("rag", "default")
# 注意 Jinja2 格式支持双花括号 {{context}}
prompt_template = PromptTemplate.from_template(raw_template, template_format="jinja2")

# 将动态拼接好的真实文章小抄，和用户提问，一起砸进模板的填空区里！
final_prompt = prompt_template.format(context=retrieved_context, question=user_query)

print("\n[INFO] 2. 组装后的超级 Prompt 已经生成！开始发给大模型。它长这样：")
print("\n" + "▼"*40)
print(final_prompt)
print("▲"*40 + "\n")


# ====================================================================
# 第三步：实例化大模型工厂，调用智谱打字回答
# ====================================================================
from chatchat.server.utils import get_ChatOpenAI

print("[INFO] 3. 开始连网呼叫配置里的 GLM-4 大模型（确保平台有网）...")

# 从你的 model_settings 瞬间构建 GLM-4 实体
llm = get_ChatOpenAI(model_name="glm-4", temperature=0.7)

print("[INFO] ⚡️ 大模型开始思考并打字（流式输出）...\n")
print("🤖 GLM-4 解答：", end="")
# 流式输出
for chunk in llm.stream(final_prompt):
    print(chunk.content, end="", flush=True)

print("\n\n✅ [全链路终章] 知识库 RAG 的【切分 -> 向量组装 -> L2空间测距 -> Pormpt动态注入 -> LLM生成】已大圆满结题！恭喜你！")
