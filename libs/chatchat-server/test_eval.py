from chatchat.server.agent.tools_factory.rag_quality_inspector import rag_quality_inspector
print(rag_quality_inspector.run({"query": "如何安装 Chatchat", "kb_name": "samples"}))
