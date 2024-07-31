from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import os
import uuid
from langchain_groq import ChatGroq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=groq_api_key)

# Initial setup
contextualize_q_system_prompt = (
    "بالنظر إلى سجل الدردشة وأحدث سؤال من المستخدم "
    "الذي قد يشير إلى السياق في سجل الدردشة، "
    "قم بصياغة سؤال مستقل يمكن فهمه "
    "بدون سجل الدردشة. لا تجب على السؤال، "
    "فقط قم بإعادة صياغته إذا لزم الأمر، وإلا فارجع السؤال كما هو."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

system_prompt = (
    "استخدم المستندات الفريدة التالية في قسم السياق للإجابة على الاستعلام في النهاية. إذا كنت لا تعرف الإجابة، قل فقط أنك لا تعرف، ولا تحاول اختلاق إجابة. "
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

store = {}
conversational_rag_chain = None


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


class QueryModel(BaseModel):
    session_id: str
    user_input: str


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global conversational_rag_chain
    session_id = str(uuid.uuid4())  # Generate a new unique session ID

    os.makedirs("temp", exist_ok=True)
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    loader = PyPDFLoader(file_location)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    vectorstore = Chroma.from_documents(documents=splits, embedding=CohereEmbeddings())
    retriever = vectorstore.as_retriever()

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    store[session_id] = ChatMessageHistory()  # Initialize session chat history

    return JSONResponse(content={"message": "PDF processed and session initialized.", "session_id": session_id})


@app.post("/ask-question/")
async def ask_question(query: QueryModel):
    if conversational_rag_chain is None:
        return JSONResponse(content={"error": "No PDF has been uploaded yet. Please upload a PDF first."},
                            status_code=400)

    session_id = query.session_id
    user_input = query.user_input

    conversation_history = get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": user_input}, config={
            "configurable": {"session_id": session_id}}
    )
    return JSONResponse(content={"answer": response["answer"]})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
