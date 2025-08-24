import os
import asyncio
from dotenv import load_dotenv

# Load environment
load_dotenv()
print("✅ Loaded .env")

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY in .env")

print("✅ API key found")

# ✅ Import OpenAI
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=api_key)
print("✅ OpenAI Async client created")

# 🔧 LLM Wrapper: Converts AutoGen messages to OpenAI format
class OpenAIWrapper:
    def __init__(self, client, model="gpt-3.5-turbo"):
        self._client = client
        self.model = model
        self.model_info = {"model": model, "vision": False}
        print(f"✅ OpenAIWrapper initialized for model: {model}")

    async def create(self, *args, **kwargs):
        print("\n============ LLM CALL RECEIVED ============")
        print("🔍 Args:", args)
        print("🔍 Kwargs:", kwargs)
        print("===========================================")

        # Extract messages from args[0]
        if len(args) == 0:
            raise ValueError("No arguments provided")
        
        raw_messages = args[0]
        if not isinstance(raw_messages, list):
            raise ValueError("First argument must be a list of messages")

        # Convert AutoGen messages to OpenAI format
        openai_messages = []
        for msg in raw_messages:
            # Get content — most messages have .content
            content = getattr(msg, "content", "")
            if not content:
                continue

            # Map source/role
            role = "system"
            if hasattr(msg, "source") and msg.source == "user":
                role = "user"
            elif hasattr(msg, "source") and msg.source == "assistant":
                role = "assistant"

            openai_messages.append({"role": role, "content": str(content)})

        if not openai_messages:
            raise ValueError("No valid messages to send to LLM")

        print("📤 Sending to OpenAI:")
        for msg in openai_messages:
            print(f"  → {msg}")

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                **{k: v for k, v in kwargs.items() if k not in ["metadata", "cancellation_token"]}
            )
            print("✅ OpenAI responded successfully!")
            return response
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            raise


# Create wrapper
wrapped_client = Open