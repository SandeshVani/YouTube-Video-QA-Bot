"""
app.py
======
Flask web application for YouTube Q&A Bot
"""

from flask import Flask, render_template, request, jsonify
import traceback

# Import bot
try:
    from bot import YouTubeQABot
    print("✅ Bot imported successfully")
except Exception as e:
    print(f"❌ Error importing bot: {e}")
    traceback.print_exc()

# Create Flask app
app = Flask(__name__)

# Create a single global bot instance (simple approach)
bot = None


# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/chat')
def chat():
    """Chat page."""
    return render_template('chat.html')


@app.route('/load', methods=['POST'])
def load_video():
    """Load a YouTube video."""
    global bot
    
    try:
        # Get URL from request
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No data received"})
        
        youtube_url = data.get('url', '')
        
        if not youtube_url:
            return jsonify({"success": False, "error": "Please provide a URL"})
        
        print(f"\n🔗 Received URL: {youtube_url}")
        
        # Create new bot and load video
        bot = YouTubeQABot()
        result = bot.load_video(youtube_url)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error in /load: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask a question."""
    global bot
    
    try:
        # Check if bot exists
        if bot is None:
            return jsonify({
                "success": False, 
                "error": "No video loaded. Please load a video first."
            })
        
        # Get question from request
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No data received"})
        
        question = data.get('question', '')
        
        if not question:
            return jsonify({"success": False, "error": "Please provide a question"})
        
        print(f"\n❓ Received question: {question}")
        
        # Get answer
        result = bot.ask(question)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error in /ask: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/test')
def test():
    """Test endpoint to check if server is working."""
    return jsonify({"status": "ok", "message": "Server is running!"})


# ============================================
# RUN
# ============================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting YouTube Q&A Bot Server...")
    print("="*50)
    print("📍 Open: http://127.0.0.1:5000")
    print("🧪 Test: http://127.0.0.1:5000/test")
    print("="*50 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)