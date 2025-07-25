import os
import json
import gradio as gr
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone, ServerlessSpec
from rag_evaluator import RAGAnswerEvaluator, integrate_with_chatbot
from rag_accuracy_evaluator import run_accuracy_evaluation, create_sample_test_questions, RAGAccuracyEvaluator
from rag_evaluation import evaluate_rag_system
from rag_evaluation import RAGSystemEvaluator, EvaluationMetrics, evaluate_rag_system

load_dotenv()

class HoneyExpertChatbot:
    def __init__(self, pinecone_api_key: str, groq_api_key: str, pinecone_environment: str = "us-east-1-aws"):
        self.pinecone_api_key = pinecone_api_key
        self.groq_api_key = groq_api_key
        self.pinecone_environment = pinecone_environment
        self.similarity_threshold = 0.5  # Threshold for knowledge base confidence
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Groq LLM for domain classification
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-70b-8192", # you can replace it also with: deepseek-r1-distill-llama-70b #qwen-qwq-32b ..
            temperature=0.1,
            max_tokens=1000
        )
        
        self.vector_store = None
        self.qa_chain = None
        
        # Initialize memory for conversation context
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Store individual session memories (for multiple users)
        self.session_memories = {}
    
    def get_session_memory(self, session_id: str = "default") -> ConversationBufferWindowMemory:
        """Get or create memory for a specific session"""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = ConversationBufferWindowMemory(
                k=5,  # Remember last 5 exchanges
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        return self.session_memories[session_id]
    def evaluate_performance(self, ground_truth_jsonl: str, session_id: str = "eval") -> 'EvaluationMetrics':
       return evaluate_rag_system(self, ground_truth_jsonl, session_id)
    def clear_session_memory(self, session_id: str = "default"):
        """Clear memory for a specific session"""
        if session_id in self.session_memories:
            self.session_memories[session_id].clear()
        else:
            self.session_memories[session_id] = ConversationBufferWindowMemory(
                k=5,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
    def is_honey_related(self, question: str) -> bool:
        """Use LLM to determine if question is honey/beekeeping related"""
        try:
            classification_prompt = f"""
You are a domain classifier. Determine if the following question is related to honey production, beekeeping, apiculture, or bee-related topics.

Question: "{question}"

Answer with "YES" ONLY if the question requires specific technical knowledge about:
- Honey production techniques and processes
- Beekeeping equipment and tools
- Bee biology and behavior specific to apiculture  
- Hive management practices
- Honey extraction and processing
- Bee diseases, pests, and treatments
- Apiary setup and maintenance
- Honey quality assessment
- Beekeeping seasonal management
- Commercial honey production
- Pollination management
- Queen rearing and breeding
- Swarm management
- Honey marketing and business
- Beekeeping regulations and standards
- Honey storage and preservation
- Wax production and processing
- Royal jelly, propolis, and bee products
- Beekeeping safety practices
- Climate and seasonal beekeeping

Answer with "NO" if the question is:
- General/casual questions about bees or honey
- Questions that can be answered with common knowledge
- Simple recipe requests (unless technical honey processing)
- General health benefits of honey
- Simple definitions that don't require expert knowledge
- Non-technical questions
- Questions about other insects or animals
- General agriculture not specific to bees

Answer: """
            
            response = self.llm.invoke(classification_prompt)
            return response.content.strip().upper() == "YES"
        except Exception as e:
            print(f"Error in domain classification: {e}")
            # If classification fails, assume it's not honey-related to be safe
            return False
        
    def load_jsonl_data(self, file_path: str) -> List[Dict]:
        """Load JSONL data from file"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
            print(f"Loaded {len(data)} records from {file_path}")
            return data
        except Exception as e:
            print(f"Error loading JSONL data: {e}")
            return []
    
    def check_index_stats(self, index_name: str) -> Dict:
        """Check if index exists and has data"""
        try:
            pc = Pinecone(api_key=self.pinecone_api_key)
            existing_indexes = pc.list_indexes().names()
            
            if index_name not in existing_indexes:
                return {"exists": False, "vector_count": 0}
            
            # Get index stats
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            
            return {"exists": True, "vector_count": vector_count}
            
        except Exception as e:
            print(f"Error checking index stats: {e}")
            return {"exists": False, "vector_count": 0}
    
    def connect_to_existing_index(self, index_name: str = "honey-expert") -> bool:
        """Connect to existing Pinecone index without loading data"""
        try:
            print(f"Connecting to existing index: {index_name}")
            
            # Connect to existing index
            self.vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embeddings,
                pinecone_api_key=self.pinecone_api_key
            )
            print("Connected to existing vector store successfully!")
            return True
            
        except Exception as e:
            print(f"Error connecting to existing index: {e}")
            return False
    
    def setup_vector_store(self, jsonl_data: Optional[List[Dict]] = None, index_name: str = "honey-expert", force_upload: bool = False):
        """Setup Pinecone vector store with JSONL data"""
        try:
            # Check if index already has data
            index_stats = self.check_index_stats(index_name)
            
            if index_stats["exists"] and index_stats["vector_count"] > 0 and not force_upload:
                print(f"Index '{index_name}' already exists with {index_stats['vector_count']} vectors.")
                print("Skipping data upload. Connecting to existing index...")
                return self.connect_to_existing_index(index_name)
            
            # Only load data if we need to upload
            if jsonl_data is None:
                print("No data provided and index doesn't exist. Please provide JSONL data.")
                return False
            
            # Initialize Pinecone with new API
            pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists
            existing_indexes = pc.list_indexes().names()
            
            if index_name not in existing_indexes:
                print(f"Creating new index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=384,  # MiniLM-L6-v2 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print("Index created successfully!")
            else:
                print(f"Using existing index: {index_name}")
            
            # If we reach here, we need to upload data
            print("Preparing to upload data...")
            
            # Prepare documents in smaller batches
            documents = []
            print(f"Processing {len(jsonl_data)} records...")
            
            for i, item in enumerate(jsonl_data):
                if i % 1000 == 0:
                    print(f"Processed {i} records...")
                    
                instruction = item.get('instruction', '')
                output = item.get('output', '')
                input_text = item.get('input', '')
                
                # Combine instruction and output for better context
                content = f"Question: {instruction}\nAnswer: {output}"
                if input_text:
                    content += f"\nContext: {input_text}"
                
                # Limit content size to avoid payload issues
                if len(content) > 2000:  # Reasonable size limit
                    content = content[:2000] + "..."
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "instruction": instruction[:500] if instruction else "",  # Limit metadata size
                        "output": output[:500] if output else "",
                        "input": input_text[:500] if input_text else "",
                        "source": "honey_expert_kb"
                    }
                )
                documents.append(doc)
            
            # Upload in smaller batches to avoid size limits
            batch_size = 100  # Smaller batch size
            total_docs = len(documents)
            
            print(f"Uploading {total_docs} documents in batches of {batch_size}...")
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i+batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_docs + batch_size - 1) // batch_size
                
                print(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} docs)...")
                
                try:
                    if i == 0:  # First batch - create new vector store
                        self.vector_store = PineconeVectorStore.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            index_name=index_name,
                            pinecone_api_key=self.pinecone_api_key
                        )
                    else:  # Subsequent batches - add to existing store
                        self.vector_store.add_documents(batch)
                    
                    print(f"Batch {batch_num} uploaded successfully!")
                    
                except Exception as batch_error:
                    print(f"Error uploading batch {batch_num}: {batch_error}")
                    # Continue with next batch instead of failing completely
                    continue
            
            print(f"Vector store setup complete! Uploaded documents in batches.")
            return True
            
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_custom_prompt(self):
        """Create custom prompt template for honey industry expertise with memory context"""
        template = """
You are a honey industry expert with deep knowledge of beekeeping, honey production, and related topics.
You MUST base your answer ONLY on the provided context from the knowledge base.

Context from Knowledge Base:
{context}

Previous Conversation:
{chat_history}

Current Question: {question}

CRITICAL INSTRUCTIONS:
1. Use ONLY the information provided in the Context from Knowledge Base above
2. Do NOT add information from general knowledge that isn't in the context
3. Be direct and concise - avoid phrases like "as we've discussed before" unless there's actual conversation history
4. Keep responses practical and well-structured
5. Write in a natural, conversational tone as an experienced beekeeper
6. If you cannot answer based solely on the provided context, do NOT respond - let the system handle it
7. Do not make assumptions or add details not present in the context
8. Answer directly without unnecessary introductions or conclusions
9. Focus on the essential information from the context

Based ONLY on the context provided above, here is my response:
"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
    
    def create_non_domain_prompt(self):
        """Create prompt for non-honey related questions with memory context"""
        template = """
YYou are a honey industry expert chatbot. The user has asked a question that is not related to honey production, beekeeping, or apiculture.

Previous Conversation:
{chat_history}

Current Question: {question}

Respond politely that you specialize in honey industry topics and suggest they ask about beekeeping, honey production, or related subjects. Keep your response brief and friendly. Only reference previous conversations if there's actual relevant chat history.

Your response:
"""
        
        return PromptTemplate(
            template=template,
            input_variables=["chat_history", "question"]
        )
    
    def setup_qa_chain(self):
        """Setup the QA chain with custom prompt and memory"""
        if not self.vector_store:
            print("Vector store not initialized")
            return False
            
        try:
            custom_prompt = self.create_custom_prompt()
            
            # Use ConversationalRetrievalChain instead of RetrievalQA for memory support
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}  # Retrieve top 3 similar documents
                ),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": custom_prompt},
                return_source_documents=True,
                verbose=False
            )
            
            print("QA chain with memory setup complete")
            return True
            
        except Exception as e:
            print(f"Error setting up QA chain: {e}")
            return False
    def run_accuracy_test(self, groq_api_key: str, custom_questions: List[Dict] = None):
  
       print("🧪 Running Accuracy Evaluation...")
       metrics = run_accuracy_evaluation(self, groq_api_key, custom_questions)
       return metrics
    def get_answer_with_confidence(self, question: str, session_id: str = "default") -> Tuple[str, bool, float]:
        """Get answer and determine if it's from knowledge base or general LLM"""
        try:
            # Get session-specific memory
            session_memory = self.get_session_memory(session_id)
            
            # First check if question is honey-related
            is_domain_related = self.is_honey_related(question)
            
            if not is_domain_related:
                # Handle non-domain questions with brief redirect, including chat history
                non_domain_prompt = self.create_non_domain_prompt()
                chat_history = session_memory.chat_memory.messages if hasattr(session_memory.chat_memory, 'messages') else []
                
                # Format chat history for the prompt
                formatted_history = ""
                if chat_history and len(chat_history) > 2:
                    for msg in chat_history[-4:]:  # Last 2 exchanges (4 messages)
                        if hasattr(msg, 'content'):
                            role = "Human" if msg.type == "human" else "Assistant"
                            formatted_history += f"{role}: {msg.content}\n"
                
                response = self.llm.invoke(non_domain_prompt.format(
                    chat_history=formatted_history,
                    question=question
                ))
                
                # Add to session memory
                session_memory.chat_memory.add_user_message(question)
                session_memory.chat_memory.add_ai_message(response.content)
                
                return response.content, False, 0.0
            
            # Get similar documents with scores for domain-related questions
            docs_with_scores = self.vector_store.similarity_search_with_score(
                question, k=5,
            )
            
            # Check if we have high confidence matches
            has_relevant_context = False
            max_similarity = 0.0
            
            if docs_with_scores:
                # Convert distance to similarity (1 - distance for cosine distance)
                similarities = []
                for doc, score in docs_with_scores:
                    # For cosine distance, score is between 0 and 2, where 0 is identical
                    similarity = 1 - (score / 2)  # Convert to 0-1 scale
                    similarities.append(max(0, similarity))  # Ensure non-negative
                
                if similarities:
                    max_similarity = max(similarities)
                    has_relevant_context = max_similarity > self.similarity_threshold
            if not has_relevant_context:
                no_info_response = (
    f"I'm not certain about '{question}' specifically. While I can't confirm the details, "
    f"you might consider consulting with local beekeeping associations or experienced beekeepers in your area. "
    f"\n\nI can help with topics like:\n"
    f"• Hive management and seasonal practices\n"
    f"• Honey harvesting and processing\n"
    f"• Common bee diseases and treatments\n"
    f"• Equipment recommendations\n"
    f"• Queen rearing and swarm management\n\n"
    f"Feel free to ask about any of these specific areas!"
)
                session_memory.chat_memory.add_user_message(question)
                session_memory.chat_memory.add_ai_message(no_info_response)
            
                return no_info_response, False, max_similarity
        
            # Create a temporary conversational chain with session memory
            temp_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                memory=session_memory,
                combine_docs_chain_kwargs={"prompt": self.create_custom_prompt()},
                return_source_documents=True,
                verbose=False
            )
            
            # Get answer from conversational chain
            result = temp_chain.invoke({"question": question})
            answer = result["answer"]
            if len(answer.strip()) < 20 or "I don't have" in answer or "cannot answer" in answer.lower():
                fallback_response = (
        f"I'm not certain about the specific details of '{question}'. "
        f"One possible approach would be to consult beekeeping manuals or reach out to experienced beekeepers. "
        f"\n\nCould you try asking about a more specific aspect of beekeeping or honey production?"
    )
                session_memory.chat_memory.add_user_message(question)
                session_memory.chat_memory.add_ai_message(fallback_response)
                return fallback_response, False, max_similarity
           
            # Only add confidence note if we actually have a good answer from knowledge base
            #confidence_note = f"\n\n✅ *Expert knowledge validated (Confidence: {max_similarity:.2f})*"
            final_answer = answer
        
            return final_answer, True, max_similarity

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}\n\n⚠️ *Please try rephrasing your question.*"
            import traceback
            traceback.print_exc()
            return error_msg, False, 0.0
    def get_answer_with_confidence_with_eval(self, question: str, session_id: str = "default", evaluator=None):
        answer, from_kb, confidence = self.get_answer_with_confidence(question, session_id)
        if evaluator and from_kb:
            try:
                docs_with_scores = self.vector_store.similarity_search_with_score(question, k=3)
                contexts = [doc.page_content for doc, score in docs_with_scores]
                eval_result = evaluator.comprehensive_evaluation(
                question=question,
                answer=answer,
                retrieved_contexts=contexts,
                confidence_score=confidence,
                from_knowledge_base=from_kb
            )
                print(f"Faithfulness: {eval_result['faithfulness_score']}/5")
                print(f"Hallucination detected: {eval_result['hallucination_detected']}")
            
            except Exception as e:
              print(f"Evaluation error: {e}")
    
        return answer, from_kb, confidence
    def chat_interface(self, message: str, history: List[Dict], session_id: str = "default") -> Tuple[str, List[Dict]]:
        """Main chat interface function with memory"""
        if not message.strip():
            return "", history
        
        # Get answer with confidence
        answer, from_kb, confidence = self.get_answer_with_confidence(message, session_id)
        
        # Add to history in new format
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        
        return "", history

def initialize_chatbot():
    """Initialize the chatbot with environment variables"""
    # Get API keys from environment variables
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    
    if not PINECONE_API_KEY or not GROQ_API_KEY:
        print("Please set your PINECONE_API_KEY and GROQ_API_KEY in your .env file")
        return None
    
    chatbot = HoneyExpertChatbot(PINECONE_API_KEY, GROQ_API_KEY, PINECONE_ENVIRONMENT)
    return chatbot

def setup_chatbot_efficiently(chatbot, jsonl_file_path: str, force_upload: bool = False):
    """Efficiently setup the chatbot by checking if data exists first"""
    # Check if index already exists with data
    index_stats = chatbot.check_index_stats("honey-expert")
    
    if index_stats["exists"] and index_stats["vector_count"] > 0 and not force_upload:
        print(f"Index already exists with {index_stats['vector_count']} vectors. Skipping data loading.")
        
        # Connect to existing index without loading JSONL data
        if not chatbot.setup_vector_store():
            print("Failed to connect to existing vector store")
            return False
    else:
        print("Index doesn't exist or is empty. Loading JSONL data...")
        data = chatbot.load_jsonl_data(jsonl_file_path)
        
        if not data:
            print("No data loaded. Please check your JSONL file.")
            return False
        
        print("Setting up vector store with new data...")
        if not chatbot.setup_vector_store(data, force_upload=force_upload):
            print("Failed to setup vector store")
            return False
    
    print("Setting up QA chain...")
    if not chatbot.setup_qa_chain():
        print("Failed to setup QA chain")
        return False
    
    print("Chatbot setup complete!")
    return True

def create_gradio_interface(chatbot):
    """Create WhatsApp/Messenger-like chat interface"""
    
    custom_css = """
    /* Main container styling */
    .gradio-container {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        margin: 0;
        padding: 0;
    }
    
    /* Remove default gradio styling */
    .gradio-container .gr-box {
        border: none !important;
        background: transparent !important;
    }
    
    /* Chat header */
    .chat-header {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff;
        padding: 20px;
        border-radius: 20px 20px 0 0;
        margin-bottom: 0;
        display: flex;
        align-items: center;
        gap: 15px;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.3);
    }
    
    .chat-header h1 {
        margin: 0;
        font-size: 20px;
        font-weight: 600;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .chat-header p {
        margin: 0;
        font-size: 13px;
        color: #fef3c7;
    }
    
    .profile-pic {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        background: linear-gradient(45deg, #fbbf24, #f59e0b);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Chat container */
    .chat-container {
        background: #ffffff;
        border-radius: 0 0 20px 20px;
        overflow: hidden;
        box-shadow: 0 4px 30px rgba(245, 158, 11, 0.2);
        border: 2px solid #fbbf24;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Override Gradio's chatbot styling */
    .gradio-container .chatbot {
        background: #ffffff !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }
    
    /* Fix message container width issues */
    .gradio-container .chatbot .message-wrap {
        max-width: none !important;
        width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove yellow background from AI messages */
    .gradio-container .chatbot .message.bot {
        background: transparent !important;
    }
    
    /* Style actual message content */
    .gradio-container .chatbot .message .content {
        background: #fef9e7 !important;
        border: 1px solid #f3e8a6 !important;
        border-radius: 18px !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        max-width: 75% !important;
        color: #8b5a2b !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
        word-wrap: break-word !important;
    }
    
    /* User message styling */
    .gradio-container .chatbot .message.user .content {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: white !important;
        margin-left: auto !important;
        border-bottom-right-radius: 4px !important;
        border: none !important;
    }
    
    /* Bot message styling */
    .gradio-container .chatbot .message.bot .content {
        background: #fef9e7 !important;
        color: #8b5a2b !important;
        margin-right: auto !important;
        border-bottom-left-radius: 4px !important;
        border: 1px solid #f3e8a6 !important;
    }
    
    /* Remove avatar containers */
    .gradio-container .chatbot .avatar {
        display: none !important;
    }
    
    /* Input area */
    .input-container {
        background: #fffbeb;
        padding: 20px;
        border-top: 2px solid #f3e8a6;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .input-container input {
        background: #fefce8 !important;
        border: 2px solid #f3e8a6 !important;
        border-radius: 25px !important;
        color: #8b5a2b !important;
        padding: 15px 20px !important;
        font-size: 15px !important;
        flex: 1;
        min-width: 0;
    }
    
    .input-container input:focus {
        border-color: #f59e0b !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.2) !important;
    }
    
    .input-container input::placeholder {
        color: #a16207 !important;
    }
    
    /* Send button */
    .btn-send {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        border: none !important;
        border-radius: 50% !important;
        width: 50px !important;
        height: 50px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: white !important;
        font-size: 18px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3) !important;
    }
    
    .btn-send:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.4) !important;
    }
    
    /* Control buttons */
    .btn-clear {
        background: #ef4444 !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 8px 16px !important;
        color: white !important;
        font-size: 13px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    .btn-clear:hover {
        background: #dc2626 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Status indicators */
    .status-indicator {
        font-size: 11px;
        color: #a16207;
        margin-top: 5px;
        text-align: right;
    }
    
    /* Scroll styling */
    .gradio-container .chatbot .overflow-y-auto {
        scrollbar-width: thin;
        scrollbar-color: #f3e8a6 #fefce8;
        padding: 20px !important;
    }
    
    .gradio-container .chatbot .overflow-y-auto::-webkit-scrollbar {
        width: 8px;
    }
    
    .gradio-container .chatbot .overflow-y-auto::-webkit-scrollbar-track {
        background: #fffbeb;
    }
    
    .gradio-container .chatbot .overflow-y-auto::-webkit-scrollbar-thumb {
        background: #f3e8a6;
        border-radius: 4px;
    }
    
    .gradio-container .chatbot .overflow-y-auto::-webkit-scrollbar-thumb:hover {
        background: #eab308;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .chat-header {
            padding: 15px;
        }
        
        .input-container {
            padding: 15px;
        }
        
        .gradio-container .chatbot .message .content {
            max-width: 90% !important;
        }
        
        .input-container input {
            padding: 12px 16px !important;
            font-size: 14px !important;
        }
        
        .btn-send {
            width: 45px !important;
            height: 45px !important;
        }
    }
    
    /* Animation */
    @keyframes messageSlide {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .gradio-container .chatbot .message .content {
        animation: messageSlide 0.4s ease-out;
    }
    
    /* Clean up extra containers */
    .gradio-container .gr-form {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .gradio-container .gr-panel {
        background: transparent !important;
        border: none !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="🍯 Honey Expert Chat" ,theme=gr.themes.Soft()) as interface:
        # Chat header
        gr.HTML("""
    <div class="chat-header">
        <div class="profile-pic">🐝</div>
        <div>
            <h1>Honey Expert</h1>
            <p>Your Beekeeping Specialist</p>
        </div>
    </div>
    """)
        with gr.Column(elem_classes="chat-container"):
            chatbot_component = gr.Chatbot(
                value=[],
                height=450,
                show_copy_button=False,
                avatar_images=None,
                type="messages",
                elem_classes="chatbot"
            )
            
            # Input area
            with gr.Row(elem_classes="input-container"):
                msg = gr.Textbox(
                    placeholder="Type your beekeeping question...",
                    container=False,
                    scale=8,
                    lines=1,
                    max_lines=3
                )
                
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button("➤", elem_classes="btn-send", size="sm")
                    
            # Control buttons
            with gr.Row(elem_classes="input-container"):
                gr.HTML("""
       <div style="color: #9ca3af; font-size: 10px; text-align: right;">
        ✅ Expert Knowledge | ⚠️ General AI
         </div> """)
            with gr.Accordion("🧪 Accuracy Evaluation", open=False):
                with gr.Row():
                   eval_btn = gr.Button("Run Accuracy Test", variant="secondary")
                   eval_output = gr.Textbox(
                    label="Evaluation Results",
                    lines=10,
                    placeholder="Click 'Run Accuracy Test' to evaluate system performance"
                )
            
                def run_evaluation():
                    try:
                        groq_api_key=os.getenv("GROQ_API_KEY")
                        metrics = run_accuracy_evaluation(chatbot, groq_api_key)
                    
                        result = f"""
🎯 ACCURACY EVALUATION RESULTS
================================
Overall Accuracy: {metrics.overall_accuracy:.1%}
Correct Answers: {metrics.correct_answers}/{metrics.total_questions}

📊 Detailed Scores:
• Factual Accuracy: {metrics.factual_accuracy:.1%}
• Grounding Accuracy: {metrics.grounding_accuracy:.1%}
• Relevance Accuracy: {metrics.relevance_accuracy:.1%}

Status: {'✅ EXCELLENT' if metrics.overall_accuracy >= 0.8 else '⚠️ GOOD' if metrics.overall_accuracy >= 0.6 else '❌ NEEDS IMPROVEMENT'}
"""
                        return result
                    except Exception as e:
                       return f"❌ Evaluation failed: {str(e)}"
            
                eval_btn.click(run_evaluation, outputs=eval_output)
            # Replace the existing evaluation accordion with this:
            with gr.Accordion("🧪 RAG System Evaluation", open=False):
                with gr.Row():
                    eval_btn = gr.Button("Run Accuracy Test", variant="secondary")
                    comprehensive_eval_btn = gr.Button("Run Full RAG Evaluation", variant="primary")
    
                eval_output = gr.Textbox(
        label="Evaluation Results",
        lines=15,
        placeholder="Click evaluation buttons to test system performance"
    )
    
                def run_evaluation():
                    try:
                        groq_api_key = os.getenv("GROQ_API_KEY")
                        metrics = run_accuracy_evaluation(chatbot, groq_api_key)
            
                        result = f"""
🎯 ACCURACY EVALUATION RESULTS
================================
Overall Accuracy: {metrics.overall_accuracy:.1%}
Correct Answers: {metrics.correct_answers}/{metrics.total_questions}

📊 Detailed Scores:
- Factual Accuracy: {metrics.factual_accuracy:.1%}
- Grounding Accuracy: {metrics.grounding_accuracy:.1%}
- Relevance Accuracy: {metrics.relevance_accuracy:.1%}

Status: {'✅ EXCELLENT' if metrics.overall_accuracy >= 0.8 else '⚠️ GOOD' if metrics.overall_accuracy >= 0.6 else '❌ NEEDS IMPROVEMENT'}
"""
                        return result
                    except Exception as e:
                        return f"❌ Evaluation failed: {str(e)}"
    
                def run_comprehensive_evaluation():
                    try:
            # This uses your ground truth JSONL file for comprehensive evaluation
                     ground_truth_file = "beekeeping_data_eval.jsonl"  # Update this path
                     metrics = chatbot.evaluate_performance(ground_truth_file)
            
                     result = f"""
🔬 COMPREHENSIVE RAG EVALUATION
=================================
Total Questions Evaluated: {metrics.total_questions}

📊 Core Metrics:
- Exact Match Score: {metrics.exact_match:.1%}
- F1 Score: {metrics.f1_score:.1%}
- Semantic Similarity: {metrics.semantic_similarity:.1%}
- Entailment Accuracy: {metrics.entailment_accuracy:.1%}

📈 Performance Assessment:
- Excellent (>80%): {'✅' if metrics.f1_score > 0.8 else '❌'} F1 Score
- Good Semantic Match: {'✅' if metrics.semantic_similarity > 0.7 else '❌'} Similarity
- Factual Consistency: {'✅' if metrics.entailment_accuracy > 0.6 else '❌'} Entailment

🎯 Overall System Health: {'🟢 EXCELLENT' if metrics.f1_score > 0.8 else '🟡 GOOD' if metrics.f1_score > 0.6 else '🔴 NEEDS IMPROVEMENT'}
"""
                     return result
                    except Exception as e:
                      return f"❌ Comprehensive evaluation failed: {str(e)}"
    
    # Wire up the buttons
        eval_btn.click(run_evaluation, outputs=eval_output)
        comprehensive_eval_btn.click(run_comprehensive_evaluation, outputs=eval_output)
        # Event handlers with WhatsApp-like experience
        def respond(message, history):
            if not message.strip():
                return history, ""
            
            # Use a session ID based on the conversation
            session_id = "gradio_session"
            
            # Get answer with memory context
            answer, from_kb, confidence = chatbot.get_answer_with_confidence(message, session_id)
            
            # Format messages for WhatsApp-like appearance
            user_msg = f"<div class='message user-message'>{message}</div>"
            bot_msg = f"<div class='message bot-message'>{answer}</div>"
            
            # Add to history in new format
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": bot_msg})
            
            return history, ""
        
        
        # Handle Enter key and button clicks
        msg.submit(respond, [msg, chatbot_component], [chatbot_component, msg])
        submit_btn.click(respond, [msg, chatbot_component], [chatbot_component, msg])
    
    return interface

def main():
    """Main function to run the chatbot"""
    # Initialize chatbot
    chatbot = initialize_chatbot()
    if not chatbot:
        return
    
    # Load data efficiently (replace with your JSONL file path)
    JSONL_FILE_PATH = "beekeeping_data.jsonl"
    
    # Set force_upload=False to skip upload if data already exists
    if not setup_chatbot_efficiently(chatbot, JSONL_FILE_PATH, force_upload=False):
        print("Failed to setup chatbot. Please check your data file and API keys.")
        return
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    evaluator = RAGAnswerEvaluator(GROQ_API_KEY)
    chatbot_with_eval = integrate_with_chatbot(chatbot, evaluator)
    # Create and launch Gradio interface
    interface = create_gradio_interface(chatbot_with_eval)
    
    print("Launching Honey Industry Expert Chatbot with Memory...")
    interface.launch(
        share=True,  # Set to True if you want a public link
        server_name="0.0.0.0",
        server_port=7860
    )

if __name__ == "__main__":
    main()