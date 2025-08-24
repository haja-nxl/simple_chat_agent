import os
import asyncio
from dotenv import load_dotenv
from autogen_core import SingleThreadedAgentRuntime, MessageContext
from autogen_core.base import BaseAgent
from autogen_core.components import DefaultSubscription
from autogen_core.components._routed_agent import event
from dataclasses import dataclass
from typing import List, Any
import openai

# Load environment variables from .env file
load_dotenv()

@dataclass
class ChatMessage:
    content: str
    sender: str = "user"

class SimpleAssistantAgent(BaseAgent):
    """A simple assistant agent that uses OpenAI"""
    
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conversation_history = []
    
    @event
    async def handle_message(self, message: ChatMessage, ctx: MessageContext) -> None:
        """Handle incoming chat messages"""
        print(f"\n👤 You: {message.content}")
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message.content})
        
        try:
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant created with AutoGen."},
                    *self.conversation_history
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            print(f"🤖 Assistant: {ai_response}")
            
        except Exception as e:
            print(f"❌ Error calling OpenAI: {e}")

async def main():
    """Main function to run the AutoGen agent"""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OpenAI API key not found in .env file!")
        print("Make sure your .env file contains: OPENAI_API_KEY=your-key-here")
        return
    
    print("🚀 AutoGen Simple Assistant Agent")
    print("=" * 40)
    print("💬 Chat with your AI assistant")
    print("📝 Type your messages and press Enter")
    print("🛑 Type 'exit', 'quit', or 'bye' to stop")
    print("=" * 40)
    
    # Create runtime and agent
    runtime = SingleThreadedAgentRuntime()
    agent = SimpleAssistantAgent("assistant")
    
    # Register the agent
    await runtime.register_factory(
        type_name="SimpleAssistantAgent",
        factory=lambda: agent,
    )
    
    # Start runtime
    runtime.start()
    
    try:
        print("\n✅ Agent ready! Start chatting:")
        
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye', '']:
                print("👋 Goodbye!")
                break
            
            # Send message to agent
            message = ChatMessage(content=user_input)
            await runtime.send_message(message, agent.id)
            
            # Small delay to let the agent process
            await asyncio.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n👋 Chat ended (Ctrl+C)")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await runtime.stop()

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import openai
        print("✅ OpenAI package found")
    except ImportError:
        print("❌ OpenAI package not found. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "openai"])
        import openai
    
    # Run the main function
    asyncio.run(main())