from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory

import streamlit as st

st.set_page_config(
    page_title="-> Assignment 06",
    page_icon="👨‍💻",
)


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_container = st.chat_message("ai")
        self.message_box = self.message_container.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        if self.message_box:
            self.message_box.markdown(self.message)


def load_memory(_):
    memory = st.session_state.get("memory")
    if memory is None:
        return []
    return memory.load_memory_variables({})["history"]


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file, api_key):
    import os
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def map_docs(inputs):
    documents = inputs["documents"]
    question = inputs["question"]
    results = []
    for doc in documents:
        result = map_doc_chain.invoke(
            {"context": doc.page_content, "question": question}
        )
        results.append(result.content)
    return "\n\n".join(results)


st.title("Assignment 06")

st.markdown(
    """
    Assignment 06

    - Migrate the RAG pipeline you implemented in the previous assignments to Streamlit.
    - Implement file upload and chat history.
    - Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar
    - Using st.sidebar put a link to the Github repo with the code of your Streamlit app.
"""
)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    openai_api_key = st.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="OpenAI API Key"
    )
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

    st.write("https://github.com/newids/full-stack-gpt")

if file and openai_api_key != "":
    if "llm" not in st.session_state or st.session_state.get("api_key") != openai_api_key:
        st.session_state["api_key"] = openai_api_key
        st.session_state["llm"] = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.1,
            streaming=False,
        )
        st.session_state["memory"] = ConversationSummaryBufferMemory(
            llm=st.session_state["llm"],
            max_token_limit=120,
            return_messages=True,
        )

    llm = st.session_state["llm"]
    retriever = embed_file(file, openai_api_key)

    callback_handler = ChatCallbackHandler()

    streaming_llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-3.5-turbo",
        temperature=0.1,
        streaming=True,
        callbacks=[callback_handler],
    )

    map_chain = {
        "documents": retriever,
        "question": RunnablePassthrough(),
    } | RunnableLambda(map_docs)

    map_doc_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Use the following portion of a long document to see if any of the text is relevant to answer the question. Return any relevant text verbatim. If there is no relevant text, return : ''
                -------
                {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    map_doc_chain = map_doc_prompt | llm

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Given the following extracted parts of a long document and a question, create a final answer.
                If you don't know the answer, just say that you don't know. Don't try to make up an answer.
                Context: {context}
                """,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")

        chat_history = st.session_state["memory"].load_memory_variables({})[
            "history"]

        chain = (
            {
                "context": map_chain,
                "question": RunnablePassthrough(),
                "history": lambda _: chat_history,
            }
            | final_prompt
            | streaming_llm
        )

        response = chain.invoke(message)

        st.session_state["memory"].save_context(
            {"input": message},
            {"output": callback_handler.message}
        )
