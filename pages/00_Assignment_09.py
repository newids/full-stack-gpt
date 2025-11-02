import streamlit as st
import openai
import time
import os
from datetime import datetime
import requests
import xml.etree.ElementTree as ET


def init_openai_client(api_key):
    """Initialize OpenAI client"""
    openai.api_key = api_key
    return openai


ASSISTANT_INSTRUCTIONS = """
You are a helpful assistant specialized in Cloudflare's AI documentation including AI Gateway, Workers AI, and Vectorize.

When answering questions:
1. Use the uploaded knowledge base files to provide accurate information
2. If you can't find the answer in the documentation, clearly state that you don't know
3. Provide specific details like pricing, limits, and technical specifications when available
4. Include relevant code examples when helpful
5. Always cite the source section when referencing specific documentation

Be concise but thorough in your responses.
"""


def create_assistant_context(docs_content):
    """Create context for the assistant from documentation content"""
    return {
        "content": docs_content,
        "instructions": ASSISTANT_INSTRUCTIONS,
        "name": "Cloudflare AI Docs Assistant"
    }


def create_conversation_context():
    """Create a new conversation context"""
    return {
        "messages": [],
        "created_at": datetime.now().isoformat()
    }


def send_message(client, conversation_context, assistant_context, message_content):
    """Send a message and get response using Chat Completions API"""
    try:
        # Build conversation history
        messages = [
            {
                "role": "system",
                "content": f"{assistant_context['instructions']}\n\nKnowledge Base:\n{assistant_context.get('content', '')[:4000]}..."
            }
        ]

        # Add conversation history (last 10 messages to stay within token limits)
        recent_messages = conversation_context.get('messages', [])[-10:]
        for msg in recent_messages:
            messages.append(msg)

        # Add current user message
        messages.append({
            "role": "user",
            "content": message_content
        })

        # Call OpenAI Chat Completions
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )

        assistant_response = response.choices[0].message.content

        # Update conversation context
        conversation_context['messages'].append({
            "role": "user",
            "content": message_content
        })
        conversation_context['messages'].append({
            "role": "assistant",
            "content": assistant_response
        })

        return assistant_response

    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return None


def process_docs_content(docs_text, store_name="Cloudflare AI Docs"):
    """Process documentation content for use with Chat Completions"""
    # For the older API, we'll just return the content to be used in the system message
    # Truncate if too long to fit in context window
    max_content_length = 8000  # Leave room for conversation and system instructions

    if len(docs_text) > max_content_length:
        # Take the first part of the documentation
        truncated_content = docs_text[:max_content_length] + \
            "\n\n[Content truncated due to length limits...]"
        st.warning(
            "Documentation content was truncated to fit within API limits.")
        return truncated_content

    return docs_text


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
def load_website_content(url, is_sitemap=True):
    """Load and return website content as text"""

    # Get keywords specific to this URL
    url_keywords = get_url_keywords(url)

    if not url_keywords:
        raise ValueError(
            "No AI-related keywords found in URL. Please select a valid AI documentation source.")

    all_content = []

    if is_sitemap:
        # Filter URLs from sitemap BEFORE loading any content
        st.info(f"Filtering sitemap for keywords: {url_keywords}")
        filtered_urls = filter_sitemap_urls(url, url_keywords)

        if not filtered_urls:
            raise ValueError(
                f"No URLs found in sitemap matching keywords: {url_keywords}")

        st.info(f"Found {len(filtered_urls)} URLs matching keywords")

        # Load content from filtered URLs
        for i, doc_url in enumerate(filtered_urls[:50]):  # Limit to 50 URLs
            try:
                response = requests.get(doc_url, timeout=10)
                if response.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Remove header and footer
                    for tag in soup(["header", "footer", "nav", "script", "style"]):
                        tag.decompose()

                    content = soup.get_text(separator=' ', strip=True)
                    content = ' '.join(content.split())  # Clean whitespace

                    all_content.append(
                        f"URL: {doc_url}\n\nContent:\n{content}\n\n{'='*50}\n\n")

                    if i % 10 == 0:
                        st.info(f"Loaded {i+1}/{len(filtered_urls)} pages...")
            except Exception as e:
                st.warning(f"Failed to load {doc_url}: {e}")
                continue
    else:
        # Load single page
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove header and footer
                for tag in soup(["header", "footer", "nav", "script", "style"]):
                    tag.decompose()

                content = soup.get_text(separator=' ', strip=True)
                content = ' '.join(content.split())  # Clean whitespace

                all_content.append(f"URL: {url}\n\nContent:\n{content}")
        except Exception as e:
            raise ValueError(f"Failed to load {url}: {e}")

    if not all_content:
        raise ValueError(
            f"No content loaded. Please select a different documentation source.")

    return '\n'.join(all_content)


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
        "https://github.com/newids/full-stack-gpt/blob/main/pages/00_Assignment_09.py")

    if st.button("Clear Conversation"):
        for key in ['thread_id', 'conversation_history', 'assistant_id', 'assistant_context', 'conversation_context']:
            if key in st.session_state:
                st.session_state.pop(key)
        st.rerun()


# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'assistant_id' not in st.session_state:
    st.session_state.assistant_id = None
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None
if 'assistant_context' not in st.session_state:
    st.session_state.assistant_context = None
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = None

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
elif url:
    try:
        # Initialize OpenAI client
        client = init_openai_client(api_key)

        # Load documentation if not already loaded
        if st.session_state.assistant_context is None:
            # Determine if URL is a sitemap or regular page
            is_sitemap = ".xml" in url

            # Show loading info
            if doc_option == "Full AI Documentation (Filtered)":
                st.info(
                    "Loading filtered AI documentation (this may take a moment)...")
            else:
                st.info(f"Loading {doc_option} documentation...")

            # Load website content
            docs_content = load_website_content(url, is_sitemap)

            # Process documentation content
            with st.spinner("Creating knowledge base..."):
                processed_content = process_docs_content(docs_content)
                st.session_state.assistant_context = create_assistant_context(
                    processed_content)

            st.success("Documentation loaded successfully!")

        # Create assistant context if not already created
        if st.session_state.assistant_id is None:
            with st.spinner("Setting up assistant..."):
                st.session_state.assistant_id = "cloudflare_docs_assistant_v1"
                st.success("Assistant ready!")

        # Create conversation context if not already created
        if st.session_state.thread_id is None:
            st.session_state.conversation_context = create_conversation_context()
            st.session_state.thread_id = "conversation_thread_v1"

        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("Conversation History")
            for i, exchange in enumerate(st.session_state.conversation_history):
                with st.container():
                    col1, col2 = st.columns([1, 6])
                    with col1:
                        st.markdown("**🙋 You:**")
                    with col2:
                        st.markdown(exchange['user'])

                    col1, col2 = st.columns([1, 6])
                    with col1:
                        st.markdown("**🤖 Assistant:**")
                    with col2:
                        st.markdown(exchange['assistant'])

                    st.divider()

        # Input for new question
        with st.form("chat_form", clear_on_submit=True):
            query = st.text_area(
                "Ask a question about Cloudflare's AI documentation:",
                placeholder="Example: What is the price per 1M input tokens of the llama-2-7b-chat-fp16 model?",
                height=100,
                key="user_input"
            )

            col1, col2 = st.columns([1, 5])
            with col1:
                submitted = st.form_submit_button("Send", type="primary")
            with col2:
                st.write("")  # Empty space for alignment

        # Test questions for reference
        with st.expander("Test Questions (Click to expand)"):
            st.markdown("""
            **Test these questions with the chatbot:**
            1. What is the price per 1M input tokens of the llama-2-7b-chat-fp16 model?
            2. What can I do with Cloudflare's AI Gateway?
            3. How many indexes can a single account have in Vectorize?
            4. How do I configure rate limiting with AI Gateway?
            5. What are the supported models in Workers AI?
            """)

        if submitted and query.strip():
            with st.spinner("Getting response from assistant..."):
                # Send message to assistant
                response = send_message(
                    client,
                    st.session_state.conversation_context,
                    st.session_state.assistant_context,
                    query
                )

                if response:
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'user': query,
                        'assistant': response,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                    # Rerun to show updated history
                    st.rerun()
                else:
                    st.error("Failed to get response from assistant.")
        elif submitted and not query.strip():
            st.warning("Please enter a question before submitting.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Try selecting a smaller documentation scope or check your API key.")

        # Reset session state on error
        for key in ['assistant_id', 'thread_id', 'assistant_context', 'conversation_context']:
            if key in st.session_state:
                st.session_state.pop(key)
else:
    st.info("Please select a documentation source from the sidebar to get started.")
