import os

import requests
import streamlit as st


API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("RAG Document Assistant")
st.caption("Intelligent Document Q&A with Grounded RAG and Web Fallback")


if "files" not in st.session_state:
    st.session_state.files = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_ready" not in st.session_state:
    st.session_state.api_ready = False


def check_api() -> bool:
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )
        return response.ok
    except requests.RequestException:
        return False


def upload_documents(uploaded_files) -> dict:
    payload = [
        (
            "files",
            (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            ),
        )
        for uploaded_file in uploaded_files
    ]

    response = requests.post(
        f"{API_URL}/documents/upload",
        files=payload,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def ask_question(question: str) -> dict:
    response = requests.post(
        f"{API_URL}/chat",
        json={"question": question},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.header("Backend")

    if check_api():
        st.success("FastAPI connected")
        st.session_state.api_ready = True
    else:
        st.error("FastAPI is not running")
        st.session_state.api_ready = False

    st.caption(f"API: {API_URL}")

    st.divider()
    st.header("Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_button = st.button(
        "Process Documents",
        use_container_width=True,
        type="primary",
    )

    if uploaded_files:
        st.write(
            f"{len(uploaded_files)} PDF(s) selected"
        )

    if process_button:
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        elif not st.session_state.api_ready:
            st.error("Start the FastAPI backend first.")
        else:
            try:
                with st.spinner(
                    "Processing documents through FastAPI..."
                ):
                    result = upload_documents(uploaded_files)

                st.session_state.files = result.get(
                    "files",
                    [],
                )
                st.session_state.messages = []

                st.success(
                    f"Processed {result.get('pages', 0)} pages "
                    f"into {result.get('chunks', 0)} chunks."
                )

            except requests.HTTPError as exc:
                try:
                    detail = exc.response.json().get(
                        "detail",
                        str(exc),
                    )
                except Exception:
                    detail = str(exc)

                st.error(
                    f"API error: {detail}"
                )

            except requests.RequestException:
                st.error(
                    "Could not connect to the FastAPI backend."
                )

            except Exception as exc:
                st.error(
                    f"Something went wrong: {exc}"
                )

    if st.session_state.files:
        st.divider()
        st.subheader("Loaded Documents")

        for filename in st.session_state.files:
            st.write(f"📄 {filename}")


if not st.session_state.files:
    st.info(
        "Start FastAPI, upload one or more PDFs, and "
        "click **Process Documents** to start."
    )

else:
    st.subheader("Ask a Question")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("source_type") == "pdf":
                st.caption("📄 Source: Uploaded PDF")

                documents = message.get(
                    "documents",
                    [],
                )

                if documents:
                    with st.expander("View PDF Sources"):
                        for document in documents:
                            st.write(
                                f"**{document['file']}** "
                                f"— Page {document['page']}"
                            )

            elif message.get("source_type") == "web":
                st.caption("🌐 Source: Web Search")

                web_sources = message.get(
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
                            url = source.get(
                                "url",
                                "",
                            )

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

            status_box.markdown(
                """
                <div style="display:inline-flex;align-items:center;gap:8px;
                font-size:15px;font-weight:600;padding:8px 12px;
                border-radius:8px;background:rgba(128,128,128,0.12);">
                    <span class="rag-dot"></span>
                    Checking documents and searching the web if needed...
                </div>

                <style>
                .rag-dot {
                    width:8px;
                    height:8px;
                    border-radius:50%;
                    background:currentColor;
                    animation:rag-blink 0.9s infinite;
                }

                @keyframes rag-blink {
                    0%, 100% { opacity:1; }
                    50% { opacity:0.2; }
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            try:
                result = ask_question(question)

                status_box.empty()

                answer = result.get(
                    "answer",
                    "No answer was returned.",
                )
                source_type = result.get(
                    "source_type"
                )

                st.markdown(answer)

                if source_type == "pdf":
                    st.caption("📄 Source: Uploaded PDF")

                elif source_type == "web":
                    st.caption("🌐 Source: Web Search")

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

            except requests.HTTPError as exc:
                status_box.empty()

                try:
                    detail = exc.response.json().get(
                        "detail",
                        str(exc),
                    )
                except Exception:
                    detail = str(exc)

                st.error(
                    f"API error: {detail}"
                )

            except requests.RequestException:
                status_box.empty()

                st.error(
                    "Could not connect to the FastAPI backend. "
                    "Make sure Uvicorn is running."
                )

            except Exception as exc:
                status_box.empty()

                st.error(
                    f"Something went wrong: {exc}"
                )
