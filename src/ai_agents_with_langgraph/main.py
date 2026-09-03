from dotenv import load_dotenv
load_dotenv()

from importlib.metadata import version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

print(f"Langchain Core: ", version("langchain-core"))
print(f"LangGraph: ", version("langgraph"))


def main():
    # Test OpenAI integration
    llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm_openai.invoke("Say 'setup complete!' in one word")
    print("OpenAI response:", response)

    # Test Anthropic integration
    llm_anthropic = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    response = llm_anthropic.invoke("Say 'setup complete!' in one word")
    print("Anthropic response:", response)

    print("setup complete!")

if __name__ == '__main__':
    main()