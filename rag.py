from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

class FinancialAdvisorBot:
    def __init__(self):
        # Initialize the LLaMA model
        self.llm = OllamaLLM(model="hf.co/sathvik123/llama3-finnace-chat-gguf")

        # Set up embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        #self.populate_chroma()

        # Data retriever
        self.vectordb = Chroma(
            persist_directory="./chroma",
            embedding_function=self.embeddings
        ).as_retriever(search_type="similarity", k=3)

        # Define memory to store conversation history
        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True,
        )

        # system prompt for bot behavior
        self.system_prompt = """
        You are a financial advisor chatbot providing clear, accurate, and concise responses to financial questions. 
        Explain concepts simply, and tailor answers to the user's query. 
        In case of follow-up questions use the conversation history as context, and avoid repetitive content in responses.
        """

        # custom prompt template
        self.custom_prompt = PromptTemplate(
            input_variables=["chat_history", "question", "context"],
            template=f"""
            {self.system_prompt}

            ### Conversation History:
            {{chat_history}}

            ### Retrieved Context:
            {{context}}

            ### User Query:
            {{question}}
            """
        )

        # Conversational Retrieval Chain with memory and prompt
        self.chatbot = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectordb,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.custom_prompt}
        )

    def populate_chroma(self):

        #Extract Data From the PDF File
        loader= DirectoryLoader('data/',
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

        documents=loader.load()

        #Split the Data into Text Chunks
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks=text_splitter.split_documents(documents)

        Chroma.from_documents(text_chunks, embedding=self.embeddings, persist_directory='./chroma')

    async def get_response(self, user_query: str):
        """Asynchronous method for processing user query through the chatbot."""
        response = self.chatbot.invoke({"question": user_query})
        return response["answer"]