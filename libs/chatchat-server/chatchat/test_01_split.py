import os
import sys

# 1. 把项目的根目录（libs/chatchat-server）强行塞进 Python 的环境变量里
# 这样就可以正常 from chatchat.xxxx 导入项目里的类了
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current_dir---",current_dir)
project_root = os.path.dirname(current_dir)  # 退回上一级也就是 chatchat-server 目录
print("project_root---",project_root)
sys.path.append(project_root)


# 2. 准备一个测试长文本（待会我们要切它）
# 随便写一段 100 多字、包含几个明显换行的中文介绍
test_content = """
北京，简称“京”，是中华人民共和国的首都、四个直辖市之一。它是全国的政治中心、文化中心、国际交往中心、科技创新中心。
北京历史悠久，文化灿烂，是首批国家历史文化名城、中国四大古都之一和世界上拥有世界文化遗产数最多的城市，3060年的建城史孕育了故宫、天坛、八达岭长城、颐和园等众多名胜古迹。
在现代经济方面，北京更是中国经济极为发达的城市。无数的互联网大厂都在这里设立了总部，无数怀揣梦想的实习生也在这里挥洒汗水。
"""

# 把这段文本写到一个临时的 txt 文件里，当作用户刚上传的文件
test_file_path = "temp_beijing.txt"
if not os.path.exists(test_file_path):
    with open(test_file_path,'w',encoding='utf-8') as f:
        f.write(test_content)
    print("环境测试准备完毕，测试文件为新建：", test_file_path)
else:
    print("测试文件已存在，直接使用：", test_file_path)


# 3. 导入 LangChain 的文本分割器
from chatchat.server.knowledge_base.utils import KnowledgeFile
# 源码中为了防止黑客攻击，[KnowledgeFile](cci:2://file:///Users/forever/Documents/%E5%AE%9E%E4%B9%A0/chatcaht/Langchain-Chatchat/libs/chatchat-server/chatchat/server/knowledge_base/utils.py:314:0-411:45) 初始化时会自动拼接系统知识库路径 (data/knowledge_base/xxx)
# 但我们现在做的是“白盒测试”，文件在当前目录下。怎么办？
# 答案是：先正常实例化骗过各种正则检测，然后强行把内部指向的路径换成我们的测试路径！

kb_file = KnowledgeFile(filename="temp_beijing.txt",knowledge_base_name="samples")
kb_file.filepath = test_file_path # 黑客式赋值：强行指向我们在第一步生成的那个本地假文件

#4.调用大模型前最重要的一步——“切分 (Chunking)”
print("\n[INFO] 正在调用底层 langchain text_splitter...")
# 我们故意把切分长度设置得很小（50个字切一刀），重叠段设为10个字。
# 真实生产环境中，chunk_size 通常在 250~500 左右。
docs = kb_file.file2text(chunk_size=50,chunk_overlap=10)

#5.查验“尸骨”阶段（看看切出来什么样）
print(f"\n 一篇短短的假文本，总共被强行切成了 {len(docs)} 个 Chunk 碎片。\n")

for i,doc in enumerate(docs):
    print(f"====[chunk{i+1}](大约{len(doc.page_content)}个字)====")
     # 核心字段 1: page_content 就是待会要存给 FAISS 向量库，并最终丢给大模型看的那段“纯文本”
    print(doc.page_content)
    # 核心字段 2: metadata 是这段文本带的“身份证”。当你后来问大模型“你参考了哪些资料”时，前端弹出来的那个红色的 [1][2] 就是根据这里的 metadata 生成的。
    print(f"\n📖 携带的身份证号 (Metadata): {doc.metadata}")
    print("=" * 60 + "\n")
