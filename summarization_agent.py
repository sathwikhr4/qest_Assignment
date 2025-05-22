from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Summarize complex text
def summarize_legal_text(text_chunks):
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    summarizer = load_summarize_chain(llm, chain_type="map_reduce")
    return summarizer.invoke(documents)