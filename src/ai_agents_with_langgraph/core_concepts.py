from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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

class Person(BaseModel):
    name: str = Field(description="person's name")
    age: int = Field(description="person's age in years")
    occupation: str = Field(description="person's occupation")


def demo_pydantic_output_parser():
    parser = PydanticOutputParser(pydantic_object=Person)
    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with 'name', 'age' and 'occupation' for: {description}"
        ).partial(format_instructions=parser.get_format_instructions())
    model = init_chat_model('gpt-4o-mini', temperature=0)

    chain = prompt | model | parser
    result = chain.invoke({"description": "A 30 year old artist named Maria"})
    print(type(result), "->", result)


class TaskExtraction(BaseModel):
    task: str = Field(description="The main task to do")
    priority: str = Field(description="high, medium or low")
    deadline: str | None = Field(description="deadline if mentioned")
    assignee: str | None = Field(description="person assigned if mentioned")


def demo_structured_output():
    structured_model = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(TaskExtraction)

    prompt = ChatPromptTemplate.from_template("Extract task information from: {text}")

    chain = prompt | structured_model
    
    texts = [
        "John needs to finish the report by Friday - it's urgent",
        "We should update the docs sometime next week",
        "Critical: Fix the login bug ASAP",
    ]

    print("Task Extractions:")
    for text in texts:
        result = chain.invoke({"text": text})
        print(f"\nInput: {text}")
        print(f"  Task: {result.task}")
        print(f"  Priority: {result.priority}")
        print(f"  Deadline: {result.deadline}")
        print(f"  Assignee: {result.assignee}")


def demo_parallel_chain():
    summarize_prompt = ChatPromptTemplate.from_template("Summarize in two sentences: {text}")
    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract 5 keywords in the following text: {text} \n return as a comma separated list"
    )
    sentiment_prompt = ChatPromptTemplate.from_template("What is the sentiment of the following text: {text}")
    model = init_chat_model('gpt-4o-mini', temperature=0)

    parser = StrOutputParser()
    analysis_chain = RunnableParallel(
        summary=summarize_prompt | model | parser, 
        keywords=keywords_prompt | model | parser, 
        sentiment=sentiment_prompt | model | parser
    )
    text = """
        The latest ChatGPT model introduces enhanced reasoning abilities, allowing it to handle complex, 
        multi-step tasks with greater accuracy. It now supports real-time interaction across text, audio, 
        and images, making conversations more dynamic and immersive. Improved personalization enables it 
        to adapt to individual communication styles and preferences seamlessly. 
        Additionally, it offers faster responses with richer context, ensuring more natural and engaging dialogue.

    """
    result = analysis_chain.invoke({'text': text})
    print(result)


def demo_passthrough_chain():

    def fake_retriever(input_dict):
        """ simulate a retrieve operation """
        return "LangChain was created by Harrison Chase in 2022."
    
    model = init_chat_model('gpt-4o-mini', temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}\n"
        "Context: {context}\n\n"
        "Answer the question based on the context."
    )

    chain = (
        RunnableParallel(
            context=RunnableLambda(fake_retriever), question=RunnablePassthrough()
        )
        | RunnableLambda(
            lambda x: {"context": x["context"], "question": x["question"]["question"]}
        )
        | prompt
        | model
        | StrOutputParser()
    )
    result = chain.invoke({"question": "Who created LangChain?"})
    print(f"Answer: {result}")


def demo_chain_branching():
    model = init_chat_model('gpt-4o-mini', temperature=0.5)
    
    code_prompt = ChatPromptTemplate.from_template(
        "You are a coding expert. Help with: {input}"
    )
    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer: {input}"
    )

    # Branching chain  based on classification
    def is_code_question(input_dict):
        classifier_prompt = ChatPromptTemplate.from_template(
                "Classify this as 'code' or 'general': {input}\nReturn only the classification."
            )
        
        classifer = classifier_prompt | model | StrOutputParser()
        
        classification = classifer.invoke(input_dict)
        code_question_yes = "code" in classification.lower()
        if code_question_yes:
            print("Code question!")
        else:
            print("Not a code question")
        return code_question_yes

    branch = RunnableBranch(
        (is_code_question, code_prompt | model | StrOutputParser()),
        general_prompt | model | StrOutputParser(),  # default branch
    )

    questions = [
        "How do I write a for loop in Python?",
        "What's the weather like today?",
    ]
    for q in questions:
        result = branch.invoke({"input": q})
        print(f"Q: {q}")
        print(f"A: {result[:100]}...\n")


if __name__ == '__main__':
    # uncomment one at a time:

    # demo_basic_chain()
    # demo_streaming_output()
    # demo_schema_inspect()
    # demo_message_types()
    # demo_basic_templates()
    # demo_few_shot_prompting()
    # demo_pydantic_output_parser()
    # demo_structured_output()
    # demo_parallel_chain()
    # demo_passthrough_chain()
    demo_chain_branching()
