import streamlit as st
import psycopg2
import bcrypt
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import pandas as pd
from io import StringIO
from docx import Document


# 🔹 Streamlit Config
st.set_page_config(page_title="ChatDocs", page_icon="📝", layout="wide")
st.title("📝 ChatDocs")

os.environ["GOOGLE_API_KEY"] = "AIzaSyCN6sMI7nRyuwgTovMrNbs2OdoOIEFAv60"

# 🔹 PostgreSQL Database Connection
def connect_db():
    return psycopg2.connect(
        dbname="neondb",
        user="neondb_owner",
        password="npg_1IeuRlwZdc8a",
        host="ep-dark-firefly-a8fjfreu-pooler.eastus2.azure.neon.tech",
        port="5432"
    )

# 🔹 Hash Passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def check_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

# 🔹 Validate User Login
def validate_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, password FROM users WHERE username=%s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user and check_password(password, user[1]):
        return user[0]
    return None

# 🔹 Register User
def register_user(username, email, password):
    hashed_pw = hash_password(password)
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_pw))
        conn.commit()
        return True
    except psycopg2.IntegrityError:
        return False
    finally:
        cursor.close()
        conn.close()

# 🔹 User Session Management
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

# 🔹 Always Show Sidebar
st.sidebar.title("📜 Chat History")

if st.session_state["user_id"]:
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT pdf_name FROM chat_history WHERE user_id = %s", (st.session_state["user_id"],))
    pdfs = cursor.fetchall()
    cursor.close()
    conn.close()

    if pdfs:
        selected_pdf = st.sidebar.selectbox("📂 Select a Document", [pdf[0] for pdf in pdfs])

        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT question, answer, timestamp FROM chat_history WHERE user_id = %s AND pdf_name = %s ORDER BY timestamp DESC",
            (st.session_state["user_id"], selected_pdf),
        )
        chats = cursor.fetchall()
        cursor.close()
        conn.close()

        if chats:
            for chat in chats:
                st.sidebar.write(f" **Q:** {chat[0]}")
                st.sidebar.write(f" **A:** {chat[1]}")
                # st.sidebar.write(f"⏰ **Time:** {chat[2]}")
                st.sidebar.write("---")
        else:
            st.sidebar.info("No chats found for this PDF.")
    else:
        st.sidebar.info("No chat history found.")

    # Add an empty space to push the button to the right
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)

    if st.button("🚪 Logout"):
        st.session_state["user_id"] = None
        st.session_state["username"] = None
        st.rerun()

    # Close the right-alignment div
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.sidebar.info("🔑 Please log in to see your chat history.")

# 🔹 Center Login Form and Make it Full Width
st.markdown(
    """
    <style>
        .block-container {
            max-width: 100% !important;
            padding: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🔹 Login/Signup UI
if not st.session_state["user_id"]:
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("🔑 Login / Sign Up")
        auth_mode = st.radio("", ["Login", "Sign Up"])

        if auth_mode == "Login":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                user_id = validate_user(username, password)
                if user_id:
                    st.session_state["user_id"] = user_id
                    st.session_state["username"] = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        elif auth_mode == "Sign Up":
            new_username = st.text_input("New Username")
            email = st.text_input("Email")
            new_password = st.text_input("New Password", type="password")
            if st.button("Sign Up"):
                if register_user(new_username, email, new_password):
                    st.success("✅ Account created! Please log in.")
                else:
                    st.error("❌ Username or email already exists.")
    st.stop()



# 🔹 Extract Text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df.to_string()

def extract_text_from_xlsx(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df.to_string()

def extract_text_from_txt(uploaded_file):
    return StringIO(uploaded_file.getvalue().decode("utf-8")).read()

uploaded_file = st.file_uploader("📂 Upload a document", type=["pdf", "docx", "csv", "xlsx", "txt"])

if uploaded_file is None:
    st.warning("⚠️ Please upload a document to proceed.")
    st.stop()

file_type = uploaded_file.type
if "pdf" in file_type:
    document_text = extract_text_from_pdf(uploaded_file)
elif "word" in file_type or "docx" in uploaded_file.name:
    document_text = extract_text_from_docx(uploaded_file)
elif "csv" in file_type:
    document_text = extract_text_from_csv(uploaded_file)
elif "excel" in file_type or "xlsx" in uploaded_file.name:
    document_text = extract_text_from_xlsx(uploaded_file)
elif "text" in file_type or "txt" in uploaded_file.name:
    document_text = extract_text_from_txt(uploaded_file)
else:
    st.error("❌ Unsupported file type.")
    st.stop()

# 🔹 Text Splitting & Embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(document_text)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_texts(texts, embeddings)

# 🔹 Question Answering
st.subheader("Ask a Question from the Docs:")
user_question = st.text_input("Your Question")
if user_question:
    docs = vector_store.similarity_search(user_question)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=user_question)

    # 🔹 Display Answer
    st.write("💡 **Answer:**", answer)

    # 🔹 Save Q&A to Database
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (user_id, pdf_name, question, answer) VALUES (%s, %s, %s, %s)",
        (st.session_state["user_id"], uploaded_file.name, user_question, answer),
    )
    conn.commit()
    cursor.close()
    conn.close()
