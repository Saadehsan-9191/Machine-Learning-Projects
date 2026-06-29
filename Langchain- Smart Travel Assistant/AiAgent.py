# %% [markdown]
# 1. Importing and installing required libraries

# %%
# !pip install streamlit
# !pip install python-dotenv
# !pip install langchain
# !pip install langchain-google-genai
# !pip install langchain-huggingface
# !pip install langchain-community

# %% [markdown]
# 2. Full code

# %%
import os
import re
from dotenv import load_dotenv
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()  # Load environment variables from the .env file into the current process

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Read the Gemini API key from environment variables

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Initialize HuggingFace embeddings model (not used directly here but ready for RAG)

llm = ChatGoogleGenerativeAI(  # Create a Gemini chat LLM instance
    model="gemini-2.5-flash",  # Use Gemini 2.5 Flash model for fast and efficient responses
    temperature=0.3,  # Set temperature for slightly creative but mostly focused answers
    api_key=GOOGLE_API_KEY  # Pass the Gemini API key explicitly to the LLM
)

memory = ConversationBufferMemory(  # Initialize conversation memory to store chat history
    memory_key="chat_history",  # Name of the memory key used inside prompts and agent
    return_messages=True  # Configure memory to return messages instead of plain text
)

def weather_tool(city: str) -> str:  # Define a simulated weather tool function that takes a city name
    """Simulated weather tool returning fake weather data."""  # Docstring explaining the purpose of the tool
    return f"The weather in {city} is sunny with a high of 25°C and a low of 15°C."  # Return a simple, hard-coded weather description

def currency_tool(currency_pair: str) -> str:  # Define a simulated currency tool function that takes a currency pair
    """Simulated currency tool returning fake conversion rate."""  # Docstring explaining the purpose of the tool
    if currency_pair.lower() == "usd/eur":  # Check if the requested pair is USD/EUR
        return "The conversion rate from USD to EUR is approximately 0.92."  # Return a fake conversion rate for USD/EUR
    elif currency_pair.lower() == "usd/cad":  # Check if the requested pair is USD/CAD
        return "The conversion rate from USD to CAD is approximately 1.35."  # Return a fake conversion rate for USD/CAD
    else:  # Handle any other currency pair
        return f"The conversion rate for {currency_pair} is approximately 1.0 (simulated)."  # Return a generic simulated conversion rate

weather_tool_langchain = Tool(  # Wrap the weather tool function as a LangChain Tool
    name="WeatherTool",  # Name of the tool used by the agent
    func=weather_tool,  # Python function that implements the tool logic
    description="Provides simulated weather information for a given city."  # Description to help the agent decide when to use this tool
)

currency_tool_langchain = Tool(  # Wrap the currency tool function as a LangChain Tool
    name="CurrencyTool",  # Name of the tool used by the agent
    func=currency_tool,  # Python function that implements the tool logic
    description="Provides simulated currency conversion rates for a given currency pair like USD/EUR."  # Description to help the agent decide when to use this tool
)

tools = [weather_tool_langchain, currency_tool_langchain]  # Create a list of tools to pass into the agent

agent = initialize_agent(  # Initialize a LangChain agent with tools, LLM, and memory
    tools=tools,  # Provide the list of tools the agent can use
    llm=llm,  # Provide the Gemini LLM instance
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Use zero-shot react description agent type for tool-using behavior
    memory=memory,  # Attach conversation buffer memory to the agent
    verbose=True  # Enable verbose mode to see internal reasoning and tool usage in logs
)

attractions_prompt = PromptTemplate(  # Define a prompt template for tourist attractions
    template="You are a smart travel assistant. Suggest 5 must-see tourist attractions in {city}.",  # Template text describing the task and using city as a variable
    input_variables=["city"]  # Declare that the prompt expects a 'city' variable
)

budget_prompt = PromptTemplate(  # Define a prompt template for budget-saving tips
    template="You are a smart travel assistant. Give 5 budget-saving tips for traveling to {city}.",  # Template text describing the task and using city as a variable
    input_variables=["city"]  # Declare that the prompt expects a 'city' variable
)

attractions_chain = attractions_prompt | llm | StrOutputParser()  # Create a chain that takes a city, runs the attractions prompt through the LLM, and parses the output as a string

budget_chain = budget_prompt | llm | StrOutputParser()  # Create a chain that takes a city, runs the budget prompt through the LLM, and parses the output as a string

parallel_recommendations_chain = RunnableParallel(  # Create a parallel runnable chain to execute attractions and budget chains simultaneously
    attractions=attractions_chain,  # Run the attractions chain in parallel under the key 'attractions'
    budget=budget_chain  # Run the budget chain in parallel under the key 'budget'
)

def detect_city_from_message(message: str) -> str | None:  # Define a helper function to detect a city name from user message
    """Very simple city detector based on common patterns."""  # Docstring explaining the purpose of the function
    patterns = [r"in\s+([A-Za-z]+)", r"to\s+([A-Za-z]+)", r"visit\s+([A-Za-z]+)"]  # Define regex patterns to capture words after 'in', 'to', or 'visit'
    for pattern in patterns:  # Loop through each pattern
        match = re.search(pattern, message, re.IGNORECASE)  # Search for the pattern in the message (case-insensitive)
        if match:  # If a match is found
            return match.group(1)  # Return the captured city name
    return None  # If no pattern matches, return None

st.set_page_config(page_title="Smart Travel Assistant", page_icon="✈️")  # Configure the Streamlit page title and icon

st.title("✈️ Smart Travel Assistant (LangChain + Gemini)")  # Display the main title of the app

st.write("Chat with a travel assistant that knows weather, currency, and can suggest attractions and budget tips.")  # Show a short description of the app

if "chat_history" not in st.session_state:  # Check if chat history is not yet stored in Streamlit session state
    st.session_state.chat_history = []  # Initialize chat history as an empty list

if "agent" not in st.session_state:  # Check if the agent is not yet stored in Streamlit session state
    st.session_state.agent = agent  # Store the initialized agent in session state

if "parallel_chain" not in st.session_state:  # Check if the parallel chain is not yet stored in session state
    st.session_state.parallel_chain = parallel_recommendations_chain  # Store the parallel recommendations chain in session state

for role, content in st.session_state.chat_history:  # Loop through each message in chat history
    with st.chat_message(role):  # Create a chat message block with the given role ('user' or 'assistant')
        st.markdown(content)  # Render the message content as Markdown

    
user_input = st.chat_input("Ask me anything about travel, weather, currency, or trip planning...")  # Create a chat input box for the user to type messages

if user_input:  # Check if the user has entered a message
    st.session_state.chat_history.append(("user", user_input))  # Append the user message to chat history

    with st.chat_message("user"):  # Display the user message in the chat interface
        st.markdown(user_input)  # Render the user message as Markdown

    city = detect_city_from_message(user_input)  # Try to detect a city name from the user message

    if city:  # If a city is detected
        with st.chat_message("assistant"):  # Create an assistant message block
            st.markdown(f"Detected that you are interested in **{city}**. Let me get attractions and budget tips...")  # Inform the user that the city was detected and parallel tasks will run
        parallel_result = st.session_state.parallel_chain.invoke({"city": city})  # Invoke the parallel chain with the detected city
        attractions_text = parallel_result["attractions"]  # Extract attractions text from the parallel result
        budget_text = parallel_result["budget"]  # Extract budget tips text from the parallel result

        combined_response = f"Here are some attractions in {city}:\n\n{attractions_text}\n\nHere are some budget tips for {city}:\n\n{budget_text}"  # Build a combined response string with attractions and budget tips

        st.session_state.chat_history.append(("assistant", combined_response))  # Append the combined assistant response to chat history

        with st.chat_message("assistant"):  # Display the combined assistant response in the chat interface
            st.markdown(combined_response)  # Render the combined response as Markdown

    else:  # If no city is detected in the user message
        agent_response = st.session_state.agent.run(user_input)  # Run the agent with the user input (automatic tool usage and memory)
        st.session_state.chat_history.append(("assistant", agent_response))  # Append the agent's response to chat history

        with st.chat_message("assistant"):  # Display the agent's response in the chat interface
            st.markdown(agent_response)  # Render the agent's response as Markdown

# %%
handle_parsing_errors=True

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


