import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit UI
st.set_page_config(page_title="LangChain: Summarize Text")
st.title("LangChain: Summarize Text from Website")
st.subheader("Enter URL")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Prompt template
prompt_template = """
Write a summary of this in 300 words:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# BUTTON SECTION
if st.button("Summarize"):

    if not groq_api_key.strip():
        st.error("Please enter Groq API Key.")
        st.stop()

    if not generic_url.strip():
        st.error("Please enter a URL.")
        st.stop()

    if not validators.url(generic_url):
        st.error("Invalid URL.")
        st.stop()

    try:
        with st.spinner("Fetching content..."):

            # ← Model initialized here only
            model = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=groq_api_key
            )

            # Load text
            loader = UnstructuredURLLoader(
    urls=[generic_url],
    headers={"User-Agent": "LangChainSummarizer/1.0"},
    ssl_verify=False
)

            docs = loader.load()
            text = " ".join([doc.page_content for doc in docs])

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)


            chain = prompt | model
            partial_summaries = []

            for chunk in chunks:
                response = chain.invoke({"text": chunk})
                partial_summaries.append(response.content)

            # -----------------------------
            # Final Summary (Reduce Step)
            # -----------------------------
            combined_summary = " ".join(partial_summaries)

            final_summary = chain.invoke(
                {"text": combined_summary}
            )

            st.success(final_summary.content)

    except Exception as e:
        st.error(f"Error: {e}")