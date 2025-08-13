# Building AI Applications Labs

This repository contains a collection of labs designed to teach various aspects of building AI applications. The labs cover a wide range of topics, from prompt engineering and basic LangChain concepts to more advanced topics like building RAG applications and deploying serverless LLMs on AWS Bedrock.

## How to Run

Please refer to the `how_to_run.md` and `how_to_run_first_time.md` files for detailed instructions on setting up your environment and running the labs.

## Labs Overview

### 1. Prompt Engineering

This lab introduces the fundamentals of prompt engineering, a crucial skill for effectively interacting with large language models (LLMs).

- **[01-Guidelines:](./Prompt-engineering/01-Guidelines.ipynb)** Learn the basic principles of writing clear and effective prompts.
- **[02-Iterative:](./Prompt-engineering/02-Iterative.ipynb)** Discover how to iteratively refine your prompts to get better responses from the LLM.
- **[03-Summarizing:](./Prompt-engineering/03-Summarizing.ipynb)** Use LLMs to summarize text, a common and powerful application.
- **[04-Inferring:](./Prompt-engineering/04-Inferring.ipynb)** Explore how LLMs can infer information and sentiment from text.
- **[05-Transforming:](./Prompt-engineering/05-Transforming.ipynb)** Learn to transform text from one format to another, such as translation or reformatting.
- **[06-Expanding:](./Prompt-engineering/06-Expanding.ipynb)** Use LLMs to expand on a topic, generating creative and detailed text.
- **[07-Chatbot:](./Prompt-engineering/07-Chatbot.ipynb)** Build a simple chatbot using the concepts learned in the previous notebooks.

### 2. LangChain

This lab dives into LangChain, a popular framework for building applications with LLMs.

- **[01-model-prompts-parsers:](./Langchain/01-model-prompts-parsers.ipynb)** Understand the core components of LangChain: models, prompts, and output parsers.
- **[02-memory:](./Langchain/02-memory.ipynb)** Learn how to add memory to your LangChain applications to maintain context in conversations.
- **[03-chain:](./Langchain/03-chain.ipynb)** Explore how to chain different components together to create more complex applications.
- **[04-QnA:](./Langchain/04-QnA.ipynb)** Build a question-answering application using your own data.
- **[05-Evaluation:](./Langchain/05-Evaluation.ipynb)** Learn how to evaluate the performance of your LangChain applications.
- **[06-functional_conversation-student:](./Langchain/06-functional_conversation-student.ipynb)** A practical exercise in building a functional conversational agent.

### 3. Chat With Your Own Data (LangChain)

This lab focuses on a powerful application of LLMs: building a chatbot that can answer questions about your own documents.

- **[Lesson1_Document_Loading:](./Chat-with-your-own-data-Langchain/Lesson1_Document_Loading.ipynb)** Learn how to load documents from various sources, including PDFs, YouTube videos, and URLs.
- **[Lesson2_Document_Splitting:](./Chat-with-your-own-data-Langchain/Lesson2_Document_Splitting.ipynb)** Understand the importance of splitting documents into smaller chunks for processing by LLMs.
- **[Lesson3_VectorStores_And_Embeddings:](./Chat-with-your-own-data-Langchain/Lesson3_VectorStores_And_Embeddings.ipynb)** Learn about vector stores and embeddings, the core technologies for semantic search.
- **[Lesson4_Retrieval:](./Chat-with-your-own-data-Langchain/Lesson4_Retrieval.ipynb)** Explore different retrieval techniques for finding the most relevant information in your documents.
- **[Lesson5_Question_Answer:](./Chat-with-your-own-data-Langchain/Lesson5_Question_Answer.ipynb)** Build a question-answering system that uses your own data as a knowledge base.
- **[Lesson6_Chat:](./Chat-with-your-own-data-Langchain/Lesson6_Chat.ipynb)** Create a conversational agent that can answer questions about your documents in a chat-like interface.

### 4. Database Agent

This lab teaches you how to build an AI agent that can interact with databases.

- **[L1_Your_First_AI_Agent:](./Database-agent/L1_Your_First_AI_Agent.ipynb)** A gentle introduction to the concept of AI agents.
- **[L2_Interacting_with_a_CSV_Data:](./Database-agent/L2_Interacting_with_a_CSV_Data.ipynb)** Learn how to use an agent to query and analyze data in a CSV file.
- **[L3_Connecting_to_a_SQL_Database:](./Database-agent/L3_Connecting_to_a_SQL_Database.ipynb)** Build an agent that can connect to a SQL database and answer questions in natural language.
- **[L4_Azure_OpenAI_Function_Calling_Feature:](./Database-agent/L4_Azure_OpenAI_Function_Calling_Feature.ipynb)** Explore the function calling feature of Azure OpenAI and how it can be used to build more powerful agents.
- **[L5_Leveraging_Assistants_API_for_SQL_Databases:](./Database-agent/L5_Leveraging_Assistants_API_for_SQL_Databases.ipynb)** Use the Assistants API to create a sophisticated agent for interacting with SQL databases.

### 5. Functions, Tools, and Agents with LangChain

This lab delves deeper into the world of AI agents, focusing on how to equip them with tools and functions to perform a wider range of tasks.

- **[Lesson1_OpenAI_function_Calling:](./Functions-Tool-Agents-Langchain/Lesson1_OpenAI_function_Calling.ipynb)** Learn the basics of OpenAI's function calling feature.
- **[Lesson2_LCEL:](./Functions-Tool-Agents-Langchain/Lesson2_LCEL.ipynb)** Explore the LangChain Expression Language (LCEL), a powerful way to compose chains and agents.
- **[Lesson3_OpenAI_function_Calling_In_Langchain:](./Functions-Tool-Agents-Langchain/Lesson3_OpenAI_function_Calling_In_Langchain.ipynb)** Integrate OpenAI's function calling feature into your LangChain applications.
- **[Lesson4__Tagging_Extraction_Using_OpenAI_functions:](./Functions-Tool-Agents-Langchain/Lesson4__Tagging_Extraction_Using_OpenAI_functions.ipynb)** Use function calling for tasks like tagging and extracting information from text.
- **[Lesson5_Tools_And_Routing:](./Functions-Tool-Agents-Langchain/Lesson5_Tools_And_Routing.ipynb)** Learn how to give your agents access to tools and how to route user requests to the appropriate tool.
- **[Lesson6_Conversational_Agent:](./Functions-Tool-Agents-Langchain/Lesson6_Conversational_Agent.ipynb)** Build a conversational agent that can use tools to answer questions and complete tasks.

### 6. LLM with Semantic Search

This lab explores the concepts and techniques behind semantic search, a powerful way to search for information based on meaning rather than just keywords.

- **[Lesson1_Keyword_Search:](./LLM_With_Semantic_Search/Lesson1_Keyword_Search.ipynb)** Start with the basics of keyword search to understand its limitations.
- **[Lesson2_Embeddings:](./LLM_With_Semantic_Search/Lesson2_Embeddings.ipynb)** Learn about embeddings, the vector representations of text that power semantic search.
- **[Lesson3_Dense_Retrieval:](./LLM_With_Semantic_Search/Lesson3_Dense_Retrieval.ipynb)** Implement dense retrieval, a core technique for semantic search.
- **[Lesson4_ReRank:](./LLM_With_Semantic_Search/Lesson4_ReRank.ipynb)** Discover how to use reranking to improve the relevance of your search results.
- **[Lesson5_Generating_Answers:](./LLM_With_Semantic_Search/Lesson5_Generating_Answers.ipynb)** Use a language model to generate natural language answers from your search results.

### 7. Agentic on Bedrock

This lab focuses on building AI agents using Amazon Bedrock, a fully managed service that makes it easy to build and scale generative AI applications.

- **[L1_Your_first_agent_with_Amazon_Bedrock:](./Agentic-on-Bedrock/L1/Lesson_1.ipynb)** A beginner's guide to creating your first AI agent with Bedrock.
- **[L2_Connecting_with_a_CRM:](./Agentic-on-Bedrock/L2/Lesson_2.ipynb)** Learn how to connect your agent to a CRM system to perform real-world tasks.
- **[L3_Performing_calculations:](./Agentic-on-Bedrock/L3/Lesson_3.ipynb)** Equip your agent with the ability to perform calculations.
- **[L4_Guard_Rails:](./Agentic-on-Bedrock/L4/Lesson_4.ipynb)** Understand how to implement guardrails to ensure your agent behaves safely and responsibly.
- **[L5_Read_the_FAQ_Manual:](./Agentic-on-Bedrock/L5/Lesson_5.ipynb)** Build an agent that can read and understand a FAQ manual to answer customer questions.

### 8. Serverless LLM on Bedrock

This lab teaches you how to build and deploy a serverless RAG (Retrieval-Augmented Generation) application using AWS Lambda and Amazon Bedrock.

- **[L1_Your_first_generations_with_Amazon_Bedrock:](./Serverless-LLM-Bedrock/L1/L1_Your_first_generations_with_Amazon_Bedrock.ipynb)** Get started with generating text using Amazon Bedrock.
- **[L2_Building_your_first_RAG_application:](./Serverless-LLM-Bedrock/L2/L2_Building_your_first_RAG_application.ipynb)** Learn the fundamentals of building a RAG application.
- **[L3_Deploying_your_RAG_application_with_Streamlit:](./Serverless-LLM-Bedrock/L3/L3_Deploying_your_RAG_application_with_Streamlit.ipynb)** Deploy your RAG application with a user-friendly interface using Streamlit.
- **[L4_Deploying_your_RAG_application_with_AWS_Lambda:](./Serverless-LLM-Bedrock/L4/L4_Deploying_your_RAG_application_with_AWS_Lambda.ipynb)** Learn how to deploy your RAG application as a serverless function on AWS Lambda.
- **[L5_Adding_conversational_memory_to_your_RAG_application:](./Serverless-LLM-Bedrock/L5/L5_Adding_conversational_memory_to_your_RAG_application.ipynb)** Add conversational memory to your RAG application to enable more natural and engaging interactions.