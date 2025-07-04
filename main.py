import os
import json
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str = Field(..., description="User's message/question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Bot's response")
    session_id: str = Field(..., description="Session ID for this conversation")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    is_expert_knowledge: bool = Field(..., description="Whether response is from expert knowledge base")
    timestamp: datetime = Field(..., description="Response timestamp")

class SessionInfo(BaseModel):
    session_id: str = Field(..., description="Session ID")
    message_count: int = Field(..., description="Number of messages in this session")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component status")

# FastAPI app
app = FastAPI(
    title="Honey Expert Chatbot API",
    description="API for honey industry expert chatbot with beekeeping knowledge",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HoneyExpertChatbot:
    def __init__(self, pinecone_api_key: str, groq_api_key: str, pinecone_environment: str = "us-east-1-aws"):
        self.pinecone_api_key = pinecone_api_key
        self.groq_api_key = groq_api_key
        self.pinecone_environment = pinecone_environment
        self.similarity_threshold = 0.3
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-70b-8192",
            temperature=0.1,
            max_tokens=1000
        )
        
        self.vector_store = None
        self.qa_chain = None
        
        # Store individual session memories and metadata
        self.session_memories = {}
        self.session_metadata = {}
        
    def get_session_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Get or create memory for a specific session"""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = ConversationBufferWindowMemory(
                k=5,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            self.session_metadata[session_id] = {
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "message_count": 0
            }
        else:
            self.session_metadata[session_id]["last_activity"] = datetime.now()
        
        return self.session_memories[session_id]
    
    def clear_session_memory(self, session_id: str):
        """Clear memory for a specific session"""
        if session_id in self.session_memories:
            self.session_memories[session_id].clear()
            self.session_metadata[session_id] = {
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "message_count": 0
            }
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        if session_id in self.session_metadata:
            return self.session_metadata[session_id]
        return None
    
    def list_sessions(self) -> List[str]:
        """List all active sessions"""
        return list(self.session_memories.keys())
        
    def is_honey_related(self, question: str) -> bool:
        """Use LLM to determine if question is honey/beekeeping related"""
        try:
            classification_prompt = f"""
You are a domain classifier. Determine if the following question is related to honey production, beekeeping, apiculture, or bee-related topics.

Question: "{question}"

Answer with only "YES" if the question is about:
- Honey production, harvesting, or processing
- Beekeeping practices and techniques
- Bee biology, behavior, or health
- Hive management and equipment
- Apiary setup and maintenance
- Bee diseases, pests, or treatments
- Pollination or bee ecology
- Honey products (wax, royal jelly, propolis, etc.)
- Honey recipes or culinary uses
- Honey market trends or economics
- Honey regulations or standards
- Honey history or cultural significance
- Honey-related research or innovations
- Beekeeping education or training
- Beekeeping community or events
- Honey conservation or environmental impact
- Honey sustainability practices
- Honey and agriculture
- Honey and nutrition
- Honey verification or quality
- Honey origins or sourcing
- Honey and health benefits
- Honey and wellness
- Honey production methods or technologies

Answer with only "NO" if the question is about unrelated topics.

Answer: """
            
            response = self.llm.invoke(classification_prompt)
            return response.content.strip().upper() == "YES"
        except Exception as e:
            print(f"Error in domain classification: {e}")
            return False
    
    def check_index_stats(self, index_name: str) -> Dict:
        """Check if index exists and has data"""
        try:
            pc = Pinecone(api_key=self.pinecone_api_key)
            existing_indexes = pc.list_indexes().names()
            
            if index_name not in existing_indexes:
                return {"exists": False, "vector_count": 0}
            
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            
            return {"exists": True, "vector_count": vector_count}
            
        except Exception as e:
            print(f"Error checking index stats: {e}")
            return {"exists": False, "vector_count": 0}
    
    def connect_to_existing_index(self, index_name: str = "honey-expert") -> bool:
        """Connect to existing Pinecone index"""
        try:
            self.vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embeddings,
                pinecone_api_key=self.pinecone_api_key
            )
            return True
        except Exception as e:
            print(f"Error connecting to existing index: {e}")
            return False
    
    def setup_vector_store(self, jsonl_data: Optional[List[Dict]] = None, index_name: str = "honey-expert", force_upload: bool = False):
        """Setup Pinecone vector store with JSONL data"""
        try:
            index_stats = self.check_index_stats(index_name)
            
            if index_stats["exists"] and index_stats["vector_count"] > 0 and not force_upload:
                print(f"Index '{index_name}' already exists with {index_stats['vector_count']} vectors.")
                return self.connect_to_existing_index(index_name)
            
            if jsonl_data is None:
                print("No data provided and index doesn't exist.")
                return False
            
            pc = Pinecone(api_key=self.pinecone_api_key)
            existing_indexes = pc.list_indexes().names()
            
            if index_name not in existing_indexes:
                pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            
            # Process documents
            documents = []
            for item in jsonl_data:
                instruction = item.get('instruction', '')
                output = item.get('output', '')
                input_text = item.get('input', '')
                
                content = f"Question: {instruction}\nAnswer: {output}"
                if input_text:
                    content += f"\nContext: {input_text}"
                
                if len(content) > 2000:
                    content = content[:2000] + "..."
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "instruction": instruction[:500] if instruction else "",
                        "output": output[:500] if output else "",
                        "input": input_text[:500] if input_text else "",
                        "source": "honey_expert_kb"
                    }
                )
                documents.append(doc)
            
            # Upload in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                if i == 0:
                    self.vector_store = PineconeVectorStore.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        index_name=index_name,
                        pinecone_api_key=self.pinecone_api_key
                    )
                else:
                    self.vector_store.add_documents(batch)
            
            return True
            
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            return False
    
    def create_custom_prompt(self):
        """Create custom prompt template"""
        template = """
You are a honey industry expert with deep knowledge of beekeeping, honey production, and related topics.

Context from Knowledge Base:
{context}

Previous Conversation:
{chat_history}

Current Question: {question}

Instructions:
1. Use conversation history for context
2. Provide concise, practical answers
3. Write in a natural, expert tone
4. Reference previous discussions when relevant
5. Keep responses focused and helpful

Your response:
"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
    
    def create_non_domain_prompt(self):
        """Create prompt for non-honey related questions"""
        template = """
You are a honey industry expert chatbot. The user asked about something outside honey/beekeeping topics.

Previous Conversation:
{chat_history}

Current Question: {question}

Politely redirect them to honey-related topics. Keep it brief and friendly.

Your response:
"""
        return PromptTemplate(
            template=template,
            input_variables=["chat_history", "question"]
        )
    
    def get_answer_with_confidence(self, question: str, session_id: str) -> Tuple[str, bool, float]:
        """Get answer with confidence score"""
        try:
            session_memory = self.get_session_memory(session_id)
            self.session_metadata[session_id]["message_count"] += 1
            
            is_domain_related = self.is_honey_related(question)
            
            if not is_domain_related:
                non_domain_prompt = self.create_non_domain_prompt()
                chat_history = session_memory.chat_memory.messages if hasattr(session_memory.chat_memory, 'messages') else []
                
                formatted_history = ""
                if chat_history:
                    for msg in chat_history[-4:]:
                        if hasattr(msg, 'content'):
                            role = "Human" if msg.type == "human" else "Assistant"
                            formatted_history += f"{role}: {msg.content}\n"
                
                response = self.llm.invoke(non_domain_prompt.format(
                    chat_history=formatted_history,
                    question=question
                ))
                
                session_memory.chat_memory.add_user_message(question)
                session_memory.chat_memory.add_ai_message(response.content)
                
                return response.content, False, 0.0
            
            # Get similar documents with scores
            docs_with_scores = self.vector_store.similarity_search_with_score(question, k=3)
            
            has_relevant_context = False
            max_similarity = 0.0
            
            if docs_with_scores:
                similarities = []
                for doc, score in docs_with_scores:
                    similarity = 1 - (score / 2)
                    similarities.append(max(0, similarity))
                
                if similarities:
                    max_similarity = max(similarities)
                    has_relevant_context = max_similarity > self.similarity_threshold
            
            # Create conversational chain
            temp_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                memory=session_memory,
                combine_docs_chain_kwargs={"prompt": self.create_custom_prompt()},
                return_source_documents=True,
                verbose=False
            )
            
            result = temp_chain.invoke({"question": question})
            answer = result["answer"]
            
            return answer, has_relevant_context, max_similarity
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}. Please try rephrasing your question."
            return error_msg, False, 0.0

# Global chatbot instance
chatbot = None

def get_chatbot():
    """Dependency to get chatbot instance"""
    global chatbot
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    return chatbot

def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load JSONL data from file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
    except Exception as e:
        print(f"Error loading JSONL data: {e}")
        return []

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    global chatbot
    
    # Get API keys from environment variables
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    
    if not PINECONE_API_KEY or not GROQ_API_KEY:
        raise Exception("Please set PINECONE_API_KEY and GROQ_API_KEY environment variables")
    
    chatbot = HoneyExpertChatbot(PINECONE_API_KEY, GROQ_API_KEY, PINECONE_ENVIRONMENT)
    
    # Setup vector store
    JSONL_FILE_PATH = "beekeeping_data.jsonl"
    index_stats = chatbot.check_index_stats("honey-expert")
    
    if index_stats["exists"] and index_stats["vector_count"] > 0:
        print(f"Connecting to existing index with {index_stats['vector_count']} vectors")
        chatbot.setup_vector_store()
    else:
        print("Loading JSONL data for new index")
        data = load_jsonl_data(JSONL_FILE_PATH)
        if data:
            chatbot.setup_vector_store(data)
        else:
            print("Warning: No data loaded, chatbot will work with general knowledge only")
    
    print("Honey Expert Chatbot API initialized successfully!")

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Honey Expert Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }



@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage, chatbot: HoneyExpertChatbot = Depends(get_chatbot)):
    """Main chat endpoint"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get response from chatbot
        response, is_expert, confidence = chatbot.get_answer_with_confidence(
            request.message, session_id
        )
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            confidence=confidence,
            is_expert_knowledge=is_expert,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str, chatbot: HoneyExpertChatbot = Depends(get_chatbot)):
    """Clear a specific session"""
    try:
        chatbot.clear_session_memory(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str, chatbot: HoneyExpertChatbot = Depends(get_chatbot)):
    """Get session information"""
    try:
        session_info = chatbot.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionInfo(
            session_id=session_id,
            message_count=session_info["message_count"],
            created_at=session_info["created_at"],
            last_activity=session_info["last_activity"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session info: {str(e)}")

@app.get("/sessions", response_model=List[str])
async def list_sessions(chatbot: HoneyExpertChatbot = Depends(get_chatbot)):
    """List all active sessions"""
    try:
        return chatbot.list_sessions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)