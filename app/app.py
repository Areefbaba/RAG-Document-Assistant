import streamlit as st
import tempfile

from rag_pipeline import (
    load_pdfs,
    split_documents,
    build_vector_store,
    get_retriever,
    answer_question,
)

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="",
    layout="wide",
)

st.title("RAG Document Assistant")
st.caption("Intelligent Document Q&A with Grounded RAG and Web Fallback")

st.markdown(
    """
    <style>
    .rag-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 15px;
        font-weight: 600;
        padding: 8px 12px;
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.12);
        margin-bottom: 8px;
    }

    .rag-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        animation: rag-blink 0.9s infinite;
    }

    @keyframes rag-blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.2; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "files" not in st.session_state:
    st.session_state.files = []

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    build_button = st.button(
        "Process Documents",
        use_container_width=True,
        type="primary",
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} PDF(s) selected")

    if build_button:
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing documents..."):
                temp_paths = []

                for uploaded_file in uploaded_files:
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf",
                    )

                    temp_file.write(uploaded_file.getbuffer())
                    temp_file.close()
                    temp_paths.append(temp_file.name)

                try:
                    documents = load_pdfs(temp_paths)

                    if not documents:
                        st.error("No readable text was found in the PDFs.")
                    else:
                        chunks = split_documents(documents)
                        vector_store = build_vector_store(chunks)
                        retriever = get_retriever(vector_store)

                        st.session_state.vector_store = vector_store
                        st.session_state.retriever = retriever
                        st.session_state.files = [
                            file.name for file in uploaded_files
                        ]
                        st.session_state.messages = []

                        st.success(
                            f"Processed {len(documents)} pages "
                            f"into {len(chunks)} chunks."
                        )

                except Exception as e:
                    st.error(f"Processing failed: {e}")

    if st.session_state.files:
        st.divider()
        st.subheader("Loaded Documents")

        for filename in st.session_state.files:
            st.write(f"{filename}")


if st.session_state.vector_store is None:
    st.info(
        "Upload one or more PDFs from the sidebar and click "
        "**Process Documents** to start."
    )

else:
    st.subheader("Ask a Question")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("source_type") == "pdf":
                st.caption("Source: Uploaded PDF")

                documents = message.get("documents", [])

                if documents:
                    with st.expander("View PDF Sources"):
                        seen = set()

                        for document in documents:
                            filename = document.metadata.get(
                                "source_file",
                                "document.pdf",
                            )

                            page = int(
                                document.metadata.get("page", 0)
                            ) + 1

                            source = (filename, page)

                            if source in seen:
                                continue

                            seen.add(source)

                            st.write(
                                f"**{filename}** — Page {page}"
                            )

            elif message.get("source_type") == "web":
                st.caption("Source: Web Search")

                web_sources = message.get("web_sources", [])

                if web_sources:
                    with st.expander("View Web Sources"):
                        for source in web_sources:
                            title = source.get(
                                "title",
                                "Web source",
                            )
                            url = source.get("url", "")

                            if url:
                                st.markdown(
                                    f"- [{title}]({url})"
                                )
                            else:
                                st.write(f"- {title}")

    question = st.chat_input(
        "Ask something about your documents..."
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            status_box = st.empty()

            def update_status(stage: str) -> None:
                status_messages = {
                    "pdf_search": "Searching your documents...",
                    "pdf_answer": "Generating a grounded answer...",
                    "web_search": "Web searching...",
                    "web_done": "Web search completed.",
                }

                status_text = status_messages.get(
                    stage,
                    "Processing...",
                )

                status_box.markdown(
                    f"""
                    <div class="rag-status">
                        <span class="rag-dot"></span>
                        {status_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            update_status("pdf_search")

            try:
                result = answer_question(
                    question=question,
                    vector_store=st.session_state.vector_store,
                    retriever=st.session_state.retriever,
                    status_callback=update_status,
                )

                answer = result["answer"]
                source_type = result["source_type"]

                status_box.empty()

                st.markdown(answer)

                if source_type == "pdf":
                    st.caption("Source: Uploaded PDF")

                    documents = result.get("documents", [])

                    if documents:
                        with st.expander("View PDF Sources"):
                            seen = set()

                            for document in documents:
                                filename = document.metadata.get(
                                    "source_file",
                                    "document.pdf",
                                )

                                page = int(
                                    document.metadata.get(
                                        "page",
                                        0,
                                    )
                                ) + 1

                                source = (filename, page)

                                if source in seen:
                                    continue

                                seen.add(source)

                                st.write(
                                    f"**{filename}** — Page {page}"
                                )

                elif source_type == "web":
                    st.caption("Source: Web Search")

                    web_sources = result.get(
                        "web_sources",
                        [],
                    )

                    if web_sources:
                        with st.expander("View Web Sources"):
                            for source in web_sources:
                                title = source.get(
                                    "title",
                                    "Web source",
                                )
                                url = source.get("url", "")

                                if url:
                                    st.markdown(
                                        f"- [{title}]({url})"
                                    )
                                else:
                                    st.write(f"- {title}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "source_type": source_type,
                        "documents": result.get(
                            "documents",
                            [],
                        ),
                        "web_sources": result.get(
                            "web_sources",
                            [],
                        ),
                    }
                )

            except Exception as e:
                status_box.empty()

                error_message = f"Something went wrong: {e}"
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
