from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

def demo_basic_chain():
    """
    Basic chain Using LCEL and Runnables
    """
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )
    model = init_chat_model(model='gpt-4o-mini', temperature=0.7)
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
    model = init_chat_model(model='gpt-4o-mini', temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    inputs = [
        {'text': 'Hello, how are you?'},
        {'text': 'What is your name?'}
    ]

    results = chain.batch(inputs)

    for text, result in zip(inputs, results):
        print(f'text: {text} => Result: {result}')

def demo_streaming_output():
    prompt = ChatPromptTemplate.from_template(
        "Write a Haiku about: {topic}"
    )
    model = init_chat_model(model='gpt-4o-mini', temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()


def demo_schema_inspect():
    prompt = ChatPromptTemplate.from_template(
            "Write a Haiku about: {topic}"
        )
    model = init_chat_model(model='gpt-4o-mini', temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"input_schema: {input_schema}")
    print(f"output_schema: {output_schema}")


def demo_message_types():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # using message objects (more control over roles)
    messages = [
        SystemMessage(content="You are a pirate. Always answer like a pirate."),
        HumanMessage(content="What's the weather like today?"),
    ]
    # print("Using message objects:")
    # print(f"Messages: {messages[0]} | {messages[1]}")

    ai_message = model.invoke(messages)
    print(f"ai_message object", ai_message)
    print(f"\nResponse from the Pirate: {ai_message.content}")

    # it's a good idea to use a bounded data structure like deque, to not overflow past context window
    # Multi-turn conversation using message objects
    messages.append(ai_message)  # add model's response to the conversation
    messages.append(HumanMessage(content="What about tomorrow?"))

    print("\nMulti-turn conversation:")
    response = model.invoke(messages)
    print(f"Follow-up response from the Pirate: {response.content}")

def demo_basic_templates():
    simple_template = ChatPromptTemplate.from_template("translate '{text}' to '{language}'")
    simple_message = simple_template.format_messages(text="Hello World", language="French")

    print("simple message from template: ", simple_message)

    multi_message_template = ChatPromptTemplate.from_messages([
        ('system', 'You are a {adjective} translator. After translating whatever the user asks to, tell a sentence according to your persona'),
        ('human', 'Translate "{text}" to "{language}" ')
    ])

    multi_messages = multi_message_template.format(adjective='grumpy old man', text="Hello World", language="French")
    print("multi messages from template: ", multi_messages)


    parser = StrOutputParser()
    model = init_chat_model('gpt-4o-mini', temperature=0)

    chain = multi_message_template | model | parser
    result = chain.invoke({'adjective': 'helpful', 'text': 'hello world', 'language': 'French'})
    print(result)


def demo_few_shot_prompting():
    examples = [
        {'word': 'happy', 'opposite': 'sad'},
        {'word': 'fast', 'opposite': 'slow'},
        {'word': 'big', 'opposite': 'small'},
    ]

    prompt_template = ChatPromptTemplate.from_messages([
        ('human', "what's the opposite of {word}?"),
        ('ai', "The opposite of {word} is {opposite}")
    ])

    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=prompt_template,
        examples=examples
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ('system', 'You give the opposite of words, follow the examples'),
        few_shot_prompt_template,
        ('human', "What's the opposite of the word {word}?")
    ])

    model = init_chat_model('gpt-4o-mini', temperature=0)
    parser = StrOutputParser()

    chain = final_prompt | model | parser
    result = chain.invoke({"word": "bright"})

    print(result)

if __name__ == '__main__':
    # uncomment one at a time:
    # demo_basic_chain()
    # demo_streaming_output()
    # demo_schema_inspect()
    # demo_message_types()
    # demo_basic_templates()
    demo_few_shot_prompting()