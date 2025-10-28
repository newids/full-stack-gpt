# Unused imports removed for efficiency
# from langchain.document_loaders import SitemapLoader
# from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st


def get_llm(api_key):
    return ChatOpenAI(
        temperature=0.1,
        openai_api_key=api_key,
    )


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    llm = inputs["llm"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata.get("source", "Unknown"),
                "date": doc.metadata.get("lastmod", "N/A"),
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    llm = inputs["llm"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


# Extract keywords from URL for targeted filtering
def get_url_keywords(url):
    """Extract relevant keywords from the URL for filtering"""
    keywords = []
    url_lower = url.lower()

    if 'ai-gateway' in url_lower:
        keywords.extend(['/ai-gateway/', 'ai-gateway'])
    if 'vectorize' in url_lower:
        keywords.extend(['/vectorize/', 'vectorize'])
    if 'workers-ai' in url_lower:
        keywords.extend(['/workers-ai/', 'workers-ai'])

    # If it's a general sitemap, include all AI keywords
    if 'sitemap' in url_lower and not keywords:
        keywords = ['/ai-gateway/', '/vectorize/', '/workers-ai/',
                    'ai-gateway', 'vectorize', 'workers-ai']

    return keywords


def filter_sitemap_urls(sitemap_url, keywords):
    """Filter URLs from sitemap XML that contain specific keywords"""
    import requests
    import xml.etree.ElementTree as ET

    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        # Handle different XML namespaces
        namespaces = {'': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        filtered_urls = []
        for url_elem in root.findall('.//loc', namespaces) or root.findall('.//loc'):
            url_text = url_elem.text
            if url_text and any(keyword in url_text.lower() for keyword in keywords):
                filtered_urls.append(url_text)
                if len(filtered_urls) >= 50:  # Limit to 50 URLs max
                    break

        return filtered_urls
    except Exception as e:
        st.error(f"Error reading sitemap: {e}")
        return []


@st.cache_data(show_spinner="Loading AI documentation...")
def load_website(url, api_key, is_sitemap=True):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,  # Reduced chunk size
        chunk_overlap=100,  # Reduced overlap
    )

    # Get keywords specific to this URL
    url_keywords = get_url_keywords(url)

    if not url_keywords:
        raise ValueError(
            "No AI-related keywords found in URL. Please select a valid AI documentation source.")

    if is_sitemap:
        # Filter URLs from sitemap BEFORE loading any content
        st.info(f"Filtering sitemap for keywords: {url_keywords}")
        filtered_urls = filter_sitemap_urls(url, url_keywords)

        if not filtered_urls:
            raise ValueError(
                f"No URLs found in sitemap matching keywords: {url_keywords}")

        st.info(f"Found {len(filtered_urls)} URLs matching keywords")

        # Load only the filtered URLs
        from langchain.document_loaders import WebBaseLoader
        loader = WebBaseLoader(filtered_urls)
        docs = loader.load_and_split(text_splitter=splitter)

        # Additional limit for safety
        if len(docs) > 200:
            docs = docs[:200]
            st.warning(
                f"Limited to first 200 documents to stay within token limits.")
    else:
        # For single page URLs - try to find related sitemap first
        from urllib.parse import urljoin, urlparse

        # Extract base domain and create sitemap URL
        parsed = urlparse(url)
        base_domain = f"{parsed.scheme}://{parsed.netloc}"

        # Try to find sitemap for the specific section
        if "/ai-gateway/" in url:
            sitemap_url = urljoin(base_domain, "/sitemap-0.xml")
        elif "/workers-ai/" in url:
            sitemap_url = urljoin(base_domain, "/sitemap-0.xml")
        elif "/vectorize/" in url:
            sitemap_url = urljoin(base_domain, "/sitemap-0.xml")
        else:
            sitemap_url = urljoin(base_domain, "/sitemap-0.xml")

        # Try to filter sitemap first
        try:
            st.info(
                f"Looking for related pages in sitemap for: {url_keywords}")
            filtered_urls = filter_sitemap_urls(sitemap_url, url_keywords)

            if filtered_urls:
                st.info(
                    f"Found {len(filtered_urls)} related URLs from sitemap")
                from langchain.document_loaders import WebBaseLoader
                # Limit to 10 related pages
                loader = WebBaseLoader(filtered_urls[:10])
                docs = loader.load_and_split(text_splitter=splitter)
            else:
                raise Exception("No related URLs found")
        except:
            # Fallback to loading just the single page
            from langchain.document_loaders import WebBaseLoader
            if any(keyword in url.lower() for keyword in url_keywords):
                loader = WebBaseLoader([url])
                docs = loader.load_and_split(text_splitter=splitter)
                st.info(f"Loaded single page")
            else:
                docs = []

    if not docs:
        raise ValueError(
            f"No documents found matching URL keywords: {url_keywords}. Please select a different documentation source.")

    vector_store = FAISS.from_documents(
        docs, OpenAIEmbeddings(openai_api_key=api_key))
    return vector_store.as_retriever()


st.set_page_config(
    page_title="Cloudflare Docs Chatbot",
    page_icon="☁️",
)


st.markdown(
    """
    # Cloudflare Docs Chatbot

    Ask questions about Cloudflare's AI documentation for:
    - AI Gateway
    - Cloudflare Vectorize
    - Workers AI

    Enter your OpenAI API key and choose a documentation sitemap from the sidebar.
"""
)


with st.sidebar:
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key to use the chatbot"
    )

    doc_option = st.selectbox(
        "Choose Documentation Source",
        [
            "",
            "Full AI Documentation (Filtered)",
            "AI Gateway Only",
            "Workers AI Only",
            "Vectorize Only"
        ],
        help="Select specific AI documentation to reduce loading time"
    )

    # Map selection to URLs
    url_mapping = {
        "Full AI Documentation (Filtered)": "https://developers.cloudflare.com/sitemap-0.xml",
        "AI Gateway Only": "https://developers.cloudflare.com/ai-gateway/",
        "Workers AI Only": "https://developers.cloudflare.com/workers-ai/",
        "Vectorize Only": "https://developers.cloudflare.com/vectorize/"
    }

    url = url_mapping.get(doc_option, "")

    st.markdown(
        "https://github.com/newid/full-stack-gpt")


if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
elif url:
    try:
        # Determine if URL is a sitemap or regular page
        is_sitemap = ".xml" in url

        # Show loading info
        if doc_option == "Full AI Documentation (Filtered)":
            st.info("Loading filtered AI documentation (this may take a moment)...")
        else:
            st.info(f"Loading {doc_option} documentation...")

        retriever = load_website(url, api_key, is_sitemap)
        llm = get_llm(api_key)

        st.success("Documentation loaded successfully!")

        query = st.text_input(
            "Ask a question about Cloudflare's AI documentation:",
            placeholder="Example: llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?"
        )

        # Test questions for reference
        with st.expander("Test Questions (Click to expand)"):
            st.markdown("""
            **Test these questions with the chatbot:**
            1. What is the price per 1M input tokens of the llama-2-7b-chat-fp16 model?
            2. What can I do with Cloudflare’s AI Gateway?
            3. How many indexes can a single account have in Vectorize?
            """)

        if query:
            with st.spinner("Searching documentation..."):
                # Get relevant documents
                docs = retriever.get_relevant_documents(query)

                # Get answers from each document
                answers_result = get_answers({
                    "docs": docs,
                    "question": query,
                    "llm": llm
                })

                # Choose the best answer
                final_result = choose_answer({
                    "answers": answers_result["answers"],
                    "question": query,
                    "llm": llm
                })

                st.markdown(final_result.content.replace("$", "\$"))
    except Exception as e:
        st.error(f"Error loading documentation: {str(e)}")
        st.info("Try selecting a smaller documentation scope or check your API key.")

        # Suggest alternatives
        if "tokens" in str(e).lower():
            st.warning(
                "Token limit exceeded. Try selecting a specific section (AI Gateway, Workers AI, or Vectorize) instead of the full documentation.")
else:
    st.info("Please select a documentation source from the sidebar to get started.")
