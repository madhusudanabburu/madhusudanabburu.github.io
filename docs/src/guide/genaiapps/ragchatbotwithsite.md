# Chat with Website - Llama3 + RAG
This section outlines the steps to create a chatbot that uses RAG (Retrieval Augmented Generation) by referencing knowledge base outside of the LLM model to generate a response to the User


## Introduction
I'm building a conversational chatbot that uses the Local Llama3 LLM model and a website's contents preloaded into a Vector database to converse with the User. As a first step, the website is identified and is split into chunks and recursively loaded into the local Vector database. This Vector database is then configured as a retrieval tool for the LLM so when a question is posed by the User, the LLM reaches out to this retriever tool to gather information

Let's start with building the application

## Prerequisites
I've used the local Llama3 as my LLM for this application, more information on installation is [here](../ossllm/llama3.md) 

## Prepare Documents for Ingestion

The first step to build an ingestion process is to identify the sources of data, I've used the below websites as my source.

- https://lilianweng.github.io/posts/2023-06-23-agent/
- https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
- https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/

These websites would be parsed using a [WebBaseLoader](https://python.langchain.com/v0.2/docs/integrations/document_loaders/web_base/). The data from the loader is actually split into multiple chunks using the [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)

The chunk size is the maximum number of characters that a chunk can contain. The chunk overlap is the number of characters that should overlap between two adjacent chunks.

I've used **chunk_size as 50** and **chunk_overlap as 10** - These values may require fine tuning

```python
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    loader = WebBaseLoader(urls, bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ))

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(docs)
```

## Ingest into Local Vector Database

Ingestion into local vector database involves initializing the ChromaDB and using [OllamaEmbeddings](https://ollama.com/blog/embedding-models) to generate vector embeddings that represent semantic meaning for a given sequence of text. This resulting vector embedding data is stored in the Vector database (I'm using Chromadb locally)

The below snippet creates the vector embeddings and also ingests the parsed documents into the collection **website_docs**

```python
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embeddings_model_name)
    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(documents=get_website_documents(), embedding=embeddings, collection_name='website_docs', persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    client = chromadb.PersistentClient(path=persist_directory, settings=CHROMA_SETTINGS)
    collection = client.get_collection(name="website_docs") 
    print("Total objects in collection - website_docs ", collection.count())
```

As shown above, the OllamaEmbeddings call uses the locally installed Ollama with the help of a model called **nomic-embed-text** to semantically encode the text 

**This is a rudimentary way of ingestion, it needs improvement in terms of how the individual documents are handled, any new changes coming from the website, how its indexed while managing the primary keys (ids) of existing records**

It also makes sense to convert this ingestion part to a seperate pipeline that can take care of ingesting the data in iteration

When the ingestion is run, it should display like below - you can see there are 3253 documents inserted into the collection. We should see our LLM parsing this collection for generating responses

![Image from images folder](~@source/images/genaiapps/ragsite/rag_agent_ingestion.png)

## Setup RAG Agent 

I've used [Langgraph](https://www.langchain.com/langgraph) to build an agent as langgraph supports better control at handling complex tasks like switching between agent and tools (often called as nodes in a graph)

Here is the node graph for this Conversational AI application

![Image from images folder](~@source/images/genaiapps/ragsite/langgraph.png)

### AgentState

Following is a class that defines the Agent state where we want to store only the messages as an initial attempt.

```
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
```

Next, we will have to define functions for calling the model, tool and also have a function **should_continue** to decide whether to continue / end and wait for the User to input a message

### Define the function that calls the model

This function is the first one to be called as the agent/model decides what action to take based on the User's question. This function will also be called once the tool response is received. The **messages** can either be from a User / tool response and can be invoked as shown below.

```python
def call_model(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}
```
Following is an example of how an agent interacts when it's called upon / posed a question

```json
{'messages': [AIMessage(content='Hello! How can I help you today?', id='run-ff95a41b-5dd9-4e00-be2e-c6f0acfdf5a0-0')]}
```
When a question is posed on **What is Task Decomposition** ? it can choose to either answer itself / reach out a tool to retrieve the answer

```json
{'messages': [AIMessage(content='', id='run-ce46d54f-0d43-4dce-b8e2-281bb5387b9e-0', tool_calls=[{'name': 'website_docs', 'args': {'query': 'Task Decomposition'}, 'id': 'call_58c7b41621ae438b85523925ad2e34fd', 'type': 'tool_call'}])]}
```

### Define the function that calls the tool

Since we are using a local Vector DB as our tool to retrieve contents, we will have to define a retriever like shown below

```python
db = Chroma(client=client, collection_name="website_docs", persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"}, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.5,
        })

tool = create_retriever_tool(
    retriever,
    "website_docs",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)

tools = [tool]
tool_executor = ToolExecutor(tools)
```

The above snippet, connects to the local Vector DB and uses the **retriever** functionality and exposes it as a tool and constructs a **ToolExecutor** object that will be used in our **call_tool** node

An important point to note when we construct the **ToolMessage** is to ensure that the **name** and the **tool_call_id** match exactly to the request from the model. This is to ensure that the agent/model understands that it has received a response from the tool for the request that was placed to it.

```python
def call_tool(state):
    messages = state["messages"]
    # We know the last message involves at least one tool call
    last_message = messages[-1]

    # We construct an ToolInvocation from the function_call
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    arguments = tool_call["args"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=arguments,
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a ToolMessage
    tool_message = ToolMessage(
        content=str(parsed_response), name=action.tool, tool_call_id=tool_call["id"]
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [tool_message]}
```

### Define the function should_continue

The **should_continue** function checks to continue only if the last call is a tool_call and there is no ToolMessage in the response

```python
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "tool_calls" not in str(last_message):
        return "end"
    # Otherwise if there is, we need to check what type of function call it is
    if "ToolMessage" in str(messages):
        return "end"
    # Otherwise we continue
    return "continue"
```

### Graph - assemble the flow between the agent, tool and any other nodes

Below is the declaration of the node graph, the starting point is the agent followed by tool as needed and then loops around depending on the User's requests

```python
    # Initialize a new graph
    graph = StateGraph(AgentState)

    # Define the two Nodes we will cycle between
    graph.add_node("agent", call_model)
    graph.add_node("action", call_tool)

    # Set the Starting Edge
    graph.set_entry_point("agent")

    # Set our Contitional Edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )

    # Set the Normal Edges
    graph.add_edge("action", "agent")

    # Compile the workflow
    agent = graph.compile()
```

## Create Chatbot UI

I've used [streamlit](https://streamlit.io/) as my UI framework to build the chat functionality. Its simple, open source and helpful in building interactive data based applications with zero to minimal front end experience

::: details Click me to view the code
```python
    # Create a side bar with information of the application - if necessary, we can collect information from the user here 
    with st.sidebar:
        st.title("📝 Conversational RAG with Website")

    st.subheader("Chatbot interaction with Local Llama3", divider="red", anchor=False)

    # Create a container to host the chat window
    message_container = st.container(height=600, border=True)

    # Declare a variable for storing the messages from the user and the agent in the session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Change the icon for the messages depending on the role - user/model
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # This is where the chat becomes interactive 
    # The messages are appended to the session state and the processing information is displayed - 'model working'
    # For every query that the user types in, the response coming out of either the agent/tool is displayed here 
    if query := st.chat_input("Enter your query here..."):
        try:
            st.session_state.messages.append(
                {"role": "user", "content": query})

            message_container.chat_message("user", avatar="😎").markdown(query)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner("model working..."):
                    inputs = {"messages": [HumanMessage(content=query)]}
                    for response in agent.stream(inputs):
                        for key, value in response.items():
                            print(f"Output from node '{key}':")
                            print("---")
                            print(value)
                            if key=='agent' or key=='action':
                                message = value["messages"][-1]
                        print("\n---\n")                        
                        # stream response
                        st.write(message.content)
                        st.session_state.messages.append({"role": "assistant", "content": message.content})

        except Exception as e:
            traceback.print_exc()
            st.error(e, icon="⛔️")
```
:::

## Demo


![Demo Video](../../../src/videos/genaiapps/ragsite/rag_agent_demo.mp4)

Following are some of the screenshots that show the interaction between **agent** and the **tool**

#### Langsmith

This image below depicts the interaction between different nodes of the application helping in evaluating and monitoring to ensure quick and confident deployments

![Image from images folder](~@source/images/genaiapps/ragsite/rag_agent_langsmith_output.png)




## Observations

Based on the demo video shown above, the following are my observations

- The agent is able to respond with correct information but the tool is still not mature to provide the right answers
    - I think the embedding model used **nomic-embed-text** may need some improvements / we need to consider a different model as the Embeddings are vectors that capture meaningful information about objects such as words or sentences
- Langsmith is a powerful tool, it helps in monitoring and debugging applications with multiple nodes and is a must have for applications written in Langgraph
- The agent/model does throw some errors at times as shown below but it still is able to continue answering the user's queries
    - ValueError: Failed to parse a response from local_Meta_Llama_LLM_8B output: {}
- The RAG based approach is a viable option to consider in closed environments where the model doesn't need to be trained but has the capability to **retrieve** from a tool there by **augmenting** its **generation** capabilities 
