import os
os.environ["USER_AGENT"] = "StreamlitApp/1.0 (teja@example.com)"
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool, AgentType
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

os.environ["GROQ_API_KEY"] = ""

def load_documents():
    try:
        loader = WebBaseLoader(web_paths=["https://travel.state.gov/content/travel/en/traveladvisories/traveladvisories.html"])
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return []

def create_vectorstore(chunks):
    if not chunks:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def setup_rag_pipeline(vectorstore):
    if vectorstore is None:
        return None
    llm = ChatGroq(model="qwen-qwq-32b", temperature=0.6)
    retriever = vectorstore.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return rag_chain

def setup_agents(rag_chain):
    if rag_chain is None:
        return None, None, None

    itinerary_tool = Tool(
        name="Itinerary Generator",
        func=rag_chain.run,
        description="Generates a trip itinerary based on user input like destination and duration."
    )

    booking_tool = Tool(
        name="Booking Searcher",
        func=lambda x: "Booking info: Flights and hotels (replace with real API call)",
        description="Searches for flights and hotels."
    )

    real_time_tool = Tool(
        name="Real-Time Assistant",
        func=lambda x: "Real-time recommendations (replace with real API call)",
        description="Provides real-time recommendations for nearby places."
    )

    llm = ChatGroq(model="qwen-qwq-32b", temperature=0.6)
    itinerary_agent = initialize_agent([itinerary_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    booking_agent = initialize_agent([booking_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    real_time_agent = initialize_agent([real_time_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return itinerary_agent, booking_agent, real_time_agent


def search_flights(origin, destination, date):
    return f"Flight search from {origin} to {destination} on {date} (replace with Skyscanner API)"

def search_hotels(city, check_in, check_out):
    return f"Hotel search in {city} from {check_in} to {check_out} (replace with HotelAPI.co)"

def get_nearby_places(location):
    return f"Nearby places for {location} (replace with Google Places API)"

def create_vectorstore(chunks):
    if not chunks:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def main():
    st.title("Trip Planning Assistant")
    st.write("Enter your trip details below to get started!")


    query = st.text_input("What would you like to plan? (e.g., 'Plan a 3-day trip to Paris')")
    
    if query:

        chunks = load_documents()
        if chunks:
            vectorstore = create_vectorstore(chunks)
            rag_chain = setup_rag_pipeline(vectorstore)
            itinerary_agent, booking_agent, real_time_agent = setup_agents(rag_chain)

            if itinerary_agent:
                with st.spinner("Planning your trip..."):
                    try:
   
                        if "trip" in query.lower() or "plan" in query.lower():
                            response = itinerary_agent.run(query)
                            st.subheader("Your Itinerary")
                            st.write(response)
                        elif "flight" in query.lower() or "book" in query.lower():
                            response = booking_agent.run(query)
                            st.subheader("Booking Information")
                            st.write(response)
                        elif "nearby" in query.lower() or "recommend" in query.lower():
                            response = real_time_agent.run(query)
                            st.subheader("Real-Time Recommendations")
                            st.write(response)
                        else:
                            response = itinerary_agent.run(query)
                            st.subheader("General Response")
                            st.write(response)
                    except Exception as e:
                        st.error(f"Error processing your request: {e}")
            else:
                st.error("Failed to initialize agents.")
        else:
            st.error("No travel data available to process your request.")

if __name__ == "__main__":
    main()
