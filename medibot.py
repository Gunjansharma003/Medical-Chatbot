import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # ✅ 1. Greeting Handling
        if prompt.lower() in ["hi", "hello", "hey"]:
            response_text = "Hello! I am MediBot 🤖. Ask me any medical question."
            st.chat_message('assistant').markdown(response_text)
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response_text
            })
            return

        # ✅ 2. Short Query Filter
        if len(prompt.split()) < 3:
            response_text = "Please ask a proper medical question."
            st.chat_message('assistant').markdown(response_text)
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response_text
            })
            return

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know and please ask a medical question only.
        Dont provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()

            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            # ✅ Working Groq model
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.0,
                api_key=os.environ.get("GROQ_API_KEY")
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={
                    'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
                }
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]

            # ✅ 3. Clean Output (no raw docs dump)
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({
                'role': 'assistant',
                'content': result
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()