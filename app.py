import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from utils_new import ChatPDFs

PDFgpt = ChatPDFs()
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat_PDFs_LangChain_App", page_icon="🐱‍💻")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your PDFs 💻")
    user_question = st.text_input("Pose a question regarding your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("My Document")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("__Processing__"):
            # get the pdf
                raw_text = PDFgpt.get_pdf_text(pdf_docs)
                # get the text chunks
                text_chunks = PDFgpt.get_txt_chunks(raw_text)
                # create vector store
                vectorstore = PDFgpt.get_vector_db(text_chunks)
                # create conversational chain
                st.session_state.conversation = PDFgpt.get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()



  # Local URL: http://localhost:8501
  # Network URL: http://192.168.29.78:8501
  #
