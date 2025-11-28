"""
bot.py
======
YouTube Q&A Bot logic
"""

import os
import warnings

warnings.filterwarnings("ignore")

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
else:
    print("⚠️ WARNING: GOOGLE_API_KEY not found in .env file!")

# YouTube API
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    print("✅ youtube-transcript-api loaded")
except ImportError as e:
    print(f"❌ Error importing youtube-transcript-api: {e}")

# LangChain components
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    print("✅ LangChain components loaded")
except ImportError as e:
    print(f"❌ Error importing LangChain: {e}")


# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_video_id(youtube_url: str) -> str:
    """Extract video ID from YouTube URL."""
    youtube_url = youtube_url.strip()
    
    if "youtu.be/" in youtube_url:
        video_id = youtube_url.split("youtu.be/")[1]
        video_id = video_id.split("?")[0]
        return video_id
    
    elif "watch?v=" in youtube_url:
        video_id = youtube_url.split("watch?v=")[1]
        video_id = video_id.split("&")[0]
        return video_id
    
    else:
        return youtube_url


def get_transcript(video_id: str) -> str:
    """Fetch transcript from YouTube."""
    ytt_api = YouTubeTranscriptApi()
    transcript_data = ytt_api.fetch(video_id)
    
    full_text = ""
    for snippet in transcript_data:
        full_text += snippet.text + " "
    
    return full_text.strip()


def create_chunks(text: str) -> list:
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def create_vector_store(chunks: list):
    """Create vector store with embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(texts=chunks, embedding=embeddings)
    return vector_store


def create_qa_chain(vector_store):
    """Create QA chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    template = """Answer based on this video context only. 
If you can't find the answer, say "I couldn't find that in the video."

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


# ============================================
# MAIN BOT CLASS
# ============================================

class YouTubeQABot:
    """Main bot class."""
    
    def __init__(self):
        self.chain = None
        self.video_id = None
        self.video_loaded = False
        print("🤖 Bot initialized")
    
    def load_video(self, youtube_url: str) -> dict:
        """Load a YouTube video."""
        print(f"\n📥 Loading video: {youtube_url}")
        
        try:
            # Step 1: Extract ID
            self.video_id = extract_video_id(youtube_url)
            print(f"   📌 Video ID: {self.video_id}")
            
            # Step 2: Get transcript
            print("   📝 Fetching transcript...")
            transcript = get_transcript(self.video_id)
            print(f"   ✅ Transcript: {len(transcript)} chars")
            
            # Step 3: Create chunks
            print("   ✂️ Creating chunks...")
            chunks = create_chunks(transcript)
            print(f"   ✅ Chunks: {len(chunks)}")
            
            # Step 4: Create vector store
            print("   🔢 Creating embeddings...")
            vector_store = create_vector_store(chunks)
            print("   ✅ Vector store ready")
            
            # Step 5: Create chain
            print("   🔗 Creating QA chain...")
            self.chain = create_qa_chain(vector_store)
            print("   ✅ Chain ready")
            
            self.video_loaded = True
            print("✅ Video loaded successfully!\n")
            
            return {
                "success": True,
                "video_id": self.video_id,
                "transcript_length": len(transcript),
                "chunks": len(chunks)
            }
        
        except Exception as e:
            print(f"❌ Error loading video: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def ask(self, question: str) -> dict:
        """Ask a question about the video."""
        print(f"\n❓ Question: {question}")
        
        if not self.video_loaded:
            return {
                "success": False,
                "error": "No video loaded. Please load a video first."
            }
        
        try:
            answer = self.chain.invoke(question)
            print(f"💡 Answer: {answer[:100]}...")
            
            return {
                "success": True,
                "question": question,
                "answer": answer
            }
        
        except Exception as e:
            print(f"❌ Error answering: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }


# Test if module loads correctly
if __name__ == "__main__":
    print("\n🧪 Testing bot.py...")
    bot = YouTubeQABot()
    print("✅ bot.py works!\n")