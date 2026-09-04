from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def demo_basic_chain():
    """
    Basic chain Using LCEL and Runnables
    """
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )
    model = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)
    parser = StrOutputParser()

    # compose a chain with pipe operator
    chain = prompt | model | parser

    result = chain.invoke({'question': 'What is LangChain?'})
    print(f'Response: {result}')

    return chain

def demo_batch_execution():
    prompt = ChatPromptTemplate.from_template(
        "Translate to French: {text}"
    )
    model = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    inputs = [
        {'text': 'Hello, how are you?'},
        {'text': 'What is your name?'}
    ]

    results = chain.batch(inputs)

    for text, result in zip(inputs, results):
        print(f'text: {text} => Result: {result}')

if __name__ == '__main__':
    # uncomment one at a time:
    # demo_basic_chain()
    demo_batch_execution()