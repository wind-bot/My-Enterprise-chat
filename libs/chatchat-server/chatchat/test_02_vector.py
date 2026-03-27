import os
import sys

# 1. 挂载环境
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
# 修复：强行指明根目录，如果不加这句，系统会找不到上一层目录里的 model_settings.yaml！
os.environ["CHATCHAT_ROOT"] = project_root

# 2. 【快速复刻前情提要】生成假数据并切成 7 块肉
from chatchat.server.knowledge_base.utils import KnowledgeFile

test_content = """
北京，简称“京”，是中华人民共和国的首都、四个直辖市之一。它是全国的政治中心、文化中心、国际交往中心、科技创新中心。
北京历史悠久，文化灿烂，是首批国家历史文化名城、中国四大古都之一和世界上拥有世界文化遗产数最多的城市，3060年的建城史孕育了故宫、天坛、八达岭长城、颐和园等众多名胜古迹。
在现代经济方面，北京更是中国经济极为发达的城市。无数的互联网大厂都在这里设立了总部，无数怀揣梦想的实习生也在这里挥洒汗水。
"""
test_file_path = "temp_beijing.txt"
if not os.path.exists(test_file_path):
    with open(test_file_path,'w',encoding='utf-8') as f:
        f.write(test_content)

kb_file = KnowledgeFile(filename="temp_beijing.txt", knowledge_base_name="samples")
kb_file.filepath = test_file_path
docs = kb_file.file2text(chunk_size=50, chunk_overlap=10)

print(f"[准备完毕] 从第一步拿到了 {len(docs)} 个已经被无损切分好的 Chunk！\n")


# 3. 【核心第二步：召唤 FAISS 知识库主管】
from chatchat.server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
from chatchat.init_database import create_tables

print("\n[INFO] 正在初始化底层的 SQLite 表 (防止 sqlite3.OperationalError)...")
create_tables()

# 我们用一个完全独立的测试名字
kb_name = "test_faiss_kb"
print(f"============================================================")
print(f"[INFO] 正在初始化 FAISS 物理向量库：{kb_name}")
print(f"============================================================")
faiss_service = FaissKBService(kb_name,embed_model="embedding-3")

# 做测试的最佳实践：查杀旧数据，从零建库
faiss_service.do_drop_kb()  
faiss_service.do_create_kb() # 这一步会在硬盘上建出真正的 index.faiss 和 index.pkl

# 4. 【高维升维与持久化】
print("\n[INFO] 开始连网呼叫 Embedding 大模型（例如智谱）...")
print("[INFO] 正在将每个带有汉字的 Chunk 翻译成 1024 维的高维数字坐标...")
print("[INFO] 正在将 [数字坐标 + 汉字 + 出处] 用胶水绑定，深深扎进 FAISS 库里...")
# 修复：防止连续调用两次 do_add_doc 导致插了两遍一样的数据！
info_doc = faiss_service.do_add_doc(docs)
print("==========成功入档的身份证名单（info_doc）:\n", info_doc)

# 5. 【见证语义搜索真正的威力】
print(f"\n============================================================")
print(f"[INFO] 我们来实机测试一下【语义检索（不靠关键字匹配）】")
print(f"============================================================")

query = "那个有很多大学刚毕业的年轻人努力打拼，流下汗水的地方具体是说什么？"
print(f"❓ [用户提问]: {query}")
print(f"（注意：这段提问里，我们故意没有讲出‘大厂’、‘实习生’、‘北京’任何一个原文核心词汇）\n")

print("[INFO] FAISS 启动！正在根据你的提问，在三维数字空间里寻找【距离最近（Cos 余弦相似度最小）】的碎片...")

# 开始执行终极搜索（绕开 Ensemble，直接查底层 FAISS 拿原始分数！）
print("[INFO] 黑客手段：正在绕过 Ensemble 的机炮外壳，直接暴力获取 FAISS 底层的距离分数...")
with faiss_service.load_vector_store().acquire() as vs:
    search_results_with_scores = vs.similarity_search_with_score(query, k=2)

print(f"\n✅ [搜索完成]！FAISS 从 7 块碎片中为你捞出了前 2 名：\n")
for i, (doc, score) in enumerate(search_results_with_scores):
    print(f"==============🥇 排名第 {i+1} 🥇==============")
    print(f"📐 距离分数: {score:.4f} （该数值是欧氏距离 L2，数值越小越匹配）")
    print(f"📖 碎片溯源: {doc.metadata.get('source')}")
    print(f"📝 碎片内容: {doc.page_content}")
    print(f"=========================================\n")
