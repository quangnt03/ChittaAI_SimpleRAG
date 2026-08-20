from langchain_core.documents import Document
from src.utils import TextDirectoryLoader
from src.app import build_application

application = build_application()
loader = TextDirectoryLoader("./data")
document = loader.load()
application.index(document)

question = "What does the source say?"
answer = application.query(question, top_k=4)

# Each stage is also available independently.
query_embedding = application.pipeline.embed_query(question)
matches = application.pipeline.search(query_embedding, top_k=4)
answer = application.pipeline.generate(question, matches)
print(matches)
print(answer)

