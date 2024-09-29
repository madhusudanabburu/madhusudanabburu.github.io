# Chat with Documents - Llama3 + RAG
This section outlines the steps to create a chatbot that uses RAG (Retrieval Augmented Generation) by referencing knowledge base outside of the LLM model to generate a response to the User


## Introduction
I'm building a conversational chatbot that uses the Local Llama3 LLM and any document's contents preloaded into a Vector database to converse with the User. As a first step, the document types are identified. I've included **PDF's, Markdown, MP3 (meeting recordings) and other TXT types**. The document is uploaded, split into chunks and recursively loaded into the local Vector database. This Vector database is then configured as a retrieval tool for the LLM chain so when a question is posed by the User, the LLM uses the retriever as its context to gather information and produce answers

Let's start with building the application

## Prerequisites
I've used the local Llama3 as my LLM for this application, more information on installation is [here](../ossllm/llama3.md) 

## Prepare Documents for Ingestion

The first step to build an ingestion process is to identify the types of data which is **PDF's, Markdown (.md), mp3/wav (meeting recordings) and TXT types**. Planning to include additional file types like **XLS, CSV, DOC, DOCX, HTML, PPT, PPTX etc**

These are the loaders which would help me in parsing the file types listed above into chunks so it can be loaded into the Vector database

- PyPDFLoader
- TextLoader
- UnstructuredMarkdownLoader,
- AssemblyAIAudioTranscriptLoader

All of the loaders listed above are available [here](https://python.langchain.com/v0.2/docs/integrations/document_loaders/). The data from the loader is split into multiple chunks using the [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)

The chunk size is the maximum number of characters that a chunk can contain. The chunk overlap is the number of characters that should overlap between two adjacent chunks.

I've used **chunk_size as 1024** and **chunk_overlap as 80** - These values may require fine tuning

```python
    # Map file extensions to document loaders and their arguments
    LOADER_MAPPING = {
        ".md": (UnstructuredMarkdownLoader, {}),
        ".pdf": (PyPDFLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"}),
        ".mp3": (AssemblyAIAudioTranscriptLoader, {})
    }

    # Check the file extension and then choose the loader type required to parse the file 
    ext = "." + file.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file, **loader_args)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(docs)
```

## Ingest into In-Memory Vector Database

Ingestion into local vector database involves initializing the [FAISS](https://github.com/facebookresearch/faiss) and using [OllamaEmbeddings](https://ollama.com/blog/embedding-models) to generate vector embeddings that represent semantic meaning for a given sequence of text. This resulting vector embedding data is stored in the Vector database (I'm using FAISS as an in-memory vector store)

The below snippet creates the vector embeddings and also ingests the parsed documents

```python
    embeddings_model_name = "mxbai-embed-large"
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embeddings_model_name)
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    db = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={},)

    # Once the loader parses the file and it's split into multiple chunks, create the vector store
    db = FAISS.from_documents(texts, embedding=embeddings)
```

As shown above, the OllamaEmbeddings call uses the locally installed Ollama with the help of a model called **mxbai-embed-large** to semantically encode the text. I chose this model from the [Huggingface MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) which ranks models based Massive Text Embedding Benchmarks. **The model was downloaded from [Ollama](https://ollama.com/search?q=mxbai)**

[HuggingFaceEmbeddings](https://python.langchain.com/v0.2/api_reference/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html#langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings) is also an alternate option to download embedding models but the challenge I had was, those were getting downloaded every time the application was instantiated which was time consuming, I prefer models to be available offline so applications can interact as needed

**This ingestion approach still needs improvement in terms of how the individual documents are handled, any parsing errors or missing sections**

## Setup RAG Agent 

### Setup Retriever

The In-memory Vector database is used as the retriever while building the conversation chain. **FAISS which is Facebook's AI for Similarity Search** is the Vector database and has high performance indexing and search capabilities. There is option to store the database locally and reload it when needed but I opted for in-memory to avoid passing unnecessary context to the LLM - only the uploaded file need to be passed in the context so the conversation is relevant

```python
    db = FAISS.from_documents(texts, embedding=embeddings)
    retriever = db.as_retriever(search_type="mmr",
            search_kwargs={
                "k": 1
            })

```

### Setup Prompts

I've provided a basic system prompt to the LLM to guide it on how to answer the user's queries. The context is going to be the parsed content from the uploaded file in the format of documents containing metadata, any url's and other text content.

```python
    ### Contextualize question ###

    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "use the following pieces of context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )   

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )


```

### Setup RAG Chain

Here, I'm instantiating the local Llama3 model by using the Ollama class provided in **langchain_community.llms** package and then creating the stuff documents chain by combining the LLM and the prompt. This chain is then passed to the retrieval chain by adding the retriever (in-memory store). The final chain created acts as the agent which answers the User's queries

```python
    ### Setup the Model ###
    llm = Ollama(model="local_Meta_Llama_LLM_8B")

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

```

## Create Chatbot UI

I've used [streamlit](https://streamlit.io/) as my UI framework to build the chat functionality. Its simple, open source and helpful in building interactive data based applications with zero to minimal front end experience

::: details Click me to view the code
```python
def side_bar():
    st.title("📝 Llama3 + Uploaded Document")
    file_uploaded = False
    progress_text = "File upload in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    global docs
    uploaded_file = st.file_uploader("'Upload a pdf/doc/html/text file' ", type=['pdf','doc','docx','html','txt', 'mp3', '.md'])
    if uploaded_file:
        my_bar.progress(1, text=progress_text)
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as file:
                file.write(uploaded_file.getvalue())
        print('file name : ', file.name)
        global db
        db = load_document(file.name, my_bar)
        file_uploaded = True
    else:
        st.write("No file was uploaded.")
    my_bar.empty()
    
    return file_uploaded

@st.experimental_fragment
def chat_window(file_uploaded):

    if file_uploaded:
        st.subheader("Chat with Documents", divider="red", anchor=False)

        react_agent = create_chain()

        message_container = st.container(height=500, border=True)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if query := st.chat_input("Enter your query here..."):
            try:
                st.session_state.messages.append(
                    {"role": "user", "content": query})

                message_container.chat_message("user", avatar="😎").markdown(query)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner("model working..."):
                        response = react_agent.invoke({"input": query})
                        print(response)
                        message = response["answer"]
                        print("message -> ", message)

                        # stream response
                        st.write(message)
                        st.session_state.messages.append({"role": "assistant", "content": message})

            except Exception as e:
                #e.with_traceback()
                st.error(e, icon="⛔️")

```
:::

## Demo

### PDF 
The following demo video shows the capability of reading, extracting and passing the PDF texts as a context to LLM to chat with it. It does work well except in a few cases where the chapters in a pdf where not identified correctly - possibly the context was missing them as there are 12 chapters in total

![Demo Video](https://github.com/madhusudanabburu/madhusudanabburu.github.io/raw/main/docs/src/videos/genaiapps/ragdocs/chat_with_documents_pdf.mp4)

### MP3

This demo is about the ability to parse an mp3 file that is related to a meeting discussion, The parsed contents are again stored in the in-memory vector store and passed as context to the LLM. The speakers, meeting discussion and the conclusions were correctly retrieved

![Demo Video](https://github.com/madhusudanabburu/madhusudanabburu.github.io/raw/main/docs/src/videos/genaiapps/ragdocs/chat_with_documents_mp3.mp4)

Here is the actual mp3 of the discussion

![Demo Audio](../../../src/videos/genaiapps/ragdocs/resources_RC_Conversation_Sample.mp3)

### Markdown

This demo is about the ability to parse a Markdown (.md) file, extract the contents and pass as context to the LLM to chat with it. It does work really well by identifying critical information but also misses some parts like the inability to identify Negative Reviews that is clearly stated in the uploaded Markdown file

![Demo Video](https://github.com/madhusudanabburu/madhusudanabburu.github.io/raw/main/docs/src/videos/genaiapps/ragdocs/chat_with_documents_md.mp4)

#### Langsmith

This image below depicts the interaction between the components of the chain consisting of LLM, Retriever etc

![Image from images folder](~@source/images/genaiapps/ragdocs/rag_langsmith_output.png)


## Sources

Available [here](https://github.com/madhusudanabburu/gen-ai-workspace/tree/main/chat-with-documents)

Please visit [LangSmith](https://smith.langchain.com/) to generate an API key and assign it to the environment variable **LANGCHAIN_API_KEY** for tracing application calls

Please visit [AssemblyAI](https://www.assemblyai.com/app/) to generate an API key and assign it to the environment variable **aai.settings.api_key** for accessing Assembly AI api's / using the **AssemblyAIAudioTranscriptLoader**


## Observations

Based on the demo videos shown above, the following are my observations

- The agent is able to respond with correct information for most of the scenarios
    - I think the embedding model used **mxbai-embed-large** is good
- Langsmith is a powerful tool, it helps in monitoring and debugging applications with multiple nodes and is a must have for applications written in Langgraph
- The RAG based approach is a viable option to consider in closed environments where the model doesn't need to be trained but has the capability to **retrieve** from a context there by **augmenting** its **generation** capabilities 
- The **create_stuff_documents_chain** and the **create_retrieval_chain** are simpler methods to create an agent for question-answering capabilities
