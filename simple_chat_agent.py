import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

class SimpleAutoGenAgent:
    """A simple AI agent using OpenAI (AutoGen-inspired)"""
    
    def __init__(self, name="Assistant"):
        self.name = name
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conversation_history = []
        self.system_message = "You are a helpful AI assistant created with AutoGen principles."
    
    def chat(self, user_message):
        """Send a message to the agent and get a response"""
        print(f"\n👤 You: {user_message}")
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_message},
                    *self.conversation_history
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            print(f"🤖 {self.name}: {ai_response}")
            return ai_response
            
        except Exception as e:
            error_msg = f"❌ Error: {e}"
            print(error_msg)
            return error_msg
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🔄 Conversation history cleared!")

def main():
    """Main function to run the chat"""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OpenAI API key not found in .env file!")
        print("Make sure your .env file contains: OPENAI_API_KEY=sk-your-key-here")
        return
    
    print("🚀 Simple AutoGen-Style Chat Agent")
    print("=" * 45)
    print("💬 Chat with your AI assistant")
    print("📝 Commands:")
    print("   - Type your message and press Enter")
    print("   - Type 'clear' to reset conversation")
    print("   - Type 'exit', 'quit', or 'bye' to stop")
    print("=" * 45)
    
    # Create the agent
    agent = SimpleAutoGenAgent("AIAssistant")
    
    try:
        print("\n✅ Agent ready! Start chatting:")
        
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                agent.reset_conversation()
                continue
            elif user_input == '':
                continue
            
            # Send message to agent
            agent.chat(user_input)
            
    except KeyboardInterrupt:
        print("\n👋 Chat ended (Ctrl+C)")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    # Check if OpenAI is installed
    try:
        import openai
        print("✅ OpenAI package ready")
    except ImportError:
        print("❌ Installing OpenAI package...")
        import subprocess
        subprocess.run(["pip", "install", "openai"])
        import openai
        print("✅ OpenAI package installed")
    
    main()