from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from RAG.agent import AgentState, GradeQuestion, GradeDocuments
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

class RAG_chatbot:
    def __init__(self):
        # Initialize the LLaMA model
        self.llm = OllamaLLM(model="hf.co/sathvik123/llama3-ChatDoc")

        # for agent
        groq_key = "Groq_KEY"
        self.groq_llm=ChatGroq(groq_api_key=groq_key, model_name="Gemma2-9b-It")

        api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1500)
        self.wiki_query=WikipediaQueryRun(api_wrapper=api_wrapper)

        # set up embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # set up chroma db
        # self.populate_chroma() 

        # data retriever
        self.retriever = Chroma(
            persist_directory="RAG/chroma",
            embedding_function=self.embeddings
        ).as_retriever(
            search_type="similarity", 
            k=5
        )

        self.store = {} # stores chat history
        
        self.get_chatbot()
         
        self.build_graph()

    def populate_chroma(self):

        #Extract Data From the PDF File
        loader= DirectoryLoader('RAG/data',
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

        documents=loader.load()

        #Split the Data into Text Chunks
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks=text_splitter.split_documents(documents)

        Chroma.from_documents(text_chunks, embedding=self.embeddings, persist_directory='RAG/chroma')

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def get_chatbot(self):

        contextualize_sys_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        contextualized_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_sys_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, contextualized_q_prompt)

        qa_sys_prompt = """You are a medical professional. \
        Use the following pieces of retrieved context to answer the question. \
        It should consist of paragraph and conversational aspect rather than just a summary. \
        If you don't know the answer, just say that you don't know. \

        {context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_sys_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_chain=create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain=create_retrieval_chain(self.history_aware_retriever, qa_chain)

        self.chatbot = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def retrieve_docs(self, state: AgentState):
        question = state["question"]
        documents = self.history_aware_retriever.invoke({"input": question, "chat_history": ""})
        state["documents"] = [doc.page_content for doc in documents]
        return state
    
    def question_classifier(self, state: AgentState):
        question = state["question"]

        system = """You are a grader assessing the topic a user question. \n
            Only answer if the question is about one of the following topics:
            1. Medical Terminology.
            2. Medical Symptoms and Diagnosis.

            Examples: What causes high blood pressure? -> Yes
                    Tell me about the CEO of Amazon -> No
                    What is tonsillitis? -> Yes
                    What is Linear Regression? -> No

            If the question IS about these topics response with "Yes", otherwise respond with "No".
            """

        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: {question}"),
            ]
        )

        structured_llm = self.groq_llm.with_structured_output(GradeQuestion)
        query_router = route_prompt | structured_llm

        result = query_router.invoke({"question": question})
        state["on_topic"] = result.score

        return state
    
    def on_topic_router(self, state: AgentState):
        on_topic = state["on_topic"]
        if on_topic.lower() == "yes":
            return "on_topic"
        return "off_topic"
    
    def web_search(self, state: AgentState):
        question = state["question"]

        state["llm_output"] = self.wiki_query.invoke({"query": question})
        return state
    
    def document_grader(self, state: AgentState):
        docs = state["documents"]
        question = state["question"]

        system = """You are a grader assessing relevance of a retrieved document to a user question. \n
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'Yes' or 'No' score to indicate whether the document is relevant to the question."""

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Retrieved document: \n\n {document} \n\n User question: {question}",
                ),
            ]
        )

        structured_llm = self.groq_llm.with_structured_output(GradeDocuments)
        grader = grade_prompt | structured_llm

        scores = []
        for doc in docs:
            result = grader.invoke({"document": doc, "question": question})
            scores.append(result.score)
        
        state["grades"] = scores
        return state
    
    def gen_router(self, state: AgentState):
        grades = state["grades"]

        if any(grade.lower() == "yes" for grade in grades):
            return "generate"
        else:
            return "rewrite_query"
        
    def rewriter(self, state: AgentState):
        question = state["question"]
        system = """You a question re-writer that converts an input question to a better version that is optimized \n
            for retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        question_rewriter = re_write_prompt | self.groq_llm | StrOutputParser()
        output = question_rewriter.invoke({"question": question})
        state["question"] = output
        return state


    def generate_answer(self, state: AgentState):
        
        question = state["question"]

        result = self.chatbot.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": "01"}
                    }, 
                )["answer"]
        
        state["llm_output"] = result
        return state
    
    def build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("topic_decision", self.question_classifier)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("retrieve_docs", self.retrieve_docs)
        workflow.add_node("rewrite_query", self.rewriter)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("document_grader", self.document_grader)

        workflow.add_edge("web_search", END)
        workflow.add_edge("retrieve_docs", "document_grader")
        workflow.add_conditional_edges(
            "topic_decision",
            self.on_topic_router,
            {
                "on_topic": "retrieve_docs",
                "off_topic": "web_search",
            },
        )
        workflow.add_conditional_edges(
            "document_grader",
            self.gen_router,
            {
                "generate": "generate_answer",
                "rewrite_query": "rewrite_query",
            },
        )
        workflow.add_edge("rewrite_query", "retrieve_docs")
        workflow.add_edge("generate_answer", END)

        workflow.set_entry_point("topic_decision")

        self.app = workflow.compile()

    async def get_response(self, user_query: str):
        """Asynchronous method for processing user query through the chatbot."""

        return self.app.invoke({"question": user_query})["llm_output"]