from langchain.docstore.document import Document
"""
class Document:
    page_content: str  # 这里装着真正的中文字：比如“感冒了应该多喝热水...”
    metadata: dict     # 这里装着这块文字的出处！比如 {"source": "常见疾病预防.pdf", "page": 12}
"""

class DocumentWithVSId(Document):
    """
    矢量化后的文档
    """

    id: str = None #底层定位id
    score: float = 3.0#相似性分数
