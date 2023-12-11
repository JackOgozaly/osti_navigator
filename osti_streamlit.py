#Used for retrieving our embeddings from google drive
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json

#General app required packages
import streamlit as st
import time
import re
from collections import Counter #Used for sorting our 
import pandas as pd

#Langchain stuff
from langchain.chains import RetrievalQAWithSourcesChain
#from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate #Used to modify the prompt for our model
from langchain.callbacks import get_openai_callback #Used to bring in stuff like cost, token usage, etc.
from langchain.embeddings import OpenAIEmbeddings #Used to embed
from langchain.vectorstores import Chroma #Used to store embeddings

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

#Streamlit customization items
st.title('OSTI Navigator')
st.caption('A LLM built on OSTI Metadata')
#Introduction text
introduction_text = "Hello! I'm OSTI naviagor bot. I can help you navigate around the OSTI repository and find useful research. How can I assist you today?"\

#Example prompts
button_1_text = 'Renewable Energy'
button_2_text = 'Solar Power'
button_3_text = 'Molten Salts'

#Bring in API Key
os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]
google_api_key = json.loads(st.secrets['google_api_key'], strict=False)

#________________________Embedding Setup_____________________________________#

#Files to download
files = ['1h7JGEiffvxPHd8TXdU7fZaEuZf093pBV', '1rP3OYZ5N5UFLcjP92ZIuLDceIk2SKKsb', 
         '1e_5xP-tbD3qW_HGgr8ax7tJ-f6b9Q-uq', '1d36ITJU0OXtfwPRrM8DO7ErzmS6DeX0P', 
         '1LYTYUK5g9FYTW49FadzyuJUBHsPReuEC', '153loCHqapm18uvcOCcLo8njRe2r_6E-t']

download_path = ['~/OSTI/',
                 '~/OSTI/3ecfa40e-85cc-4506-b101-16d7fb1eecfc/',
                 '~/OSTI/3ecfa40e-85cc-4506-b101-16d7fb1eecfc/',
                 '~/OSTI/3ecfa40e-85cc-4506-b101-16d7fb1eecfc/',
                 '~/OSTI/3ecfa40e-85cc-4506-b101-16d7fb1eecfc/',
                 '~/OSTI/3ecfa40e-85cc-4506-b101-16d7fb1eecfc/']

#Make our directory
if not os.path.exists(download_path[1]):
    os.makedirs(download_path[1])


#We only want to call the Google Drive API once per script run. Once the directory exists and has files, don't download anything
if len(os.listdir(download_path[1])) == 0:     
    # Create credentials from the JSON object
    credentials = service_account.Credentials.from_service_account_info(
             google_api_key,
             scopes=["https://www.googleapis.com/auth/drive"]
         )
    # Scope required for accessing and modifying Drive data
    #SCOPES = ['https://www.googleapis.com/auth/drive']
    
    def download_file(real_file_id, local_folder_path):
        """Downloads a file
        Args:
            real_file_id: ID of the file to download
            local_folder_path: Local path where the file will be saved
        Returns: IO object with location.
        """
       # creds = service_account.Credentials.from_service_account_file(
        #    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    
    
    
        # create drive api client
        service = build("drive", "v3", credentials=credentials)
    
        file_id = real_file_id
    
        # Get file metadata to obtain the file name
        file_metadata = service.files().get(fileId=file_id).execute()
        file_name = file_metadata['name']
    
        local_file_path = os.path.join(local_folder_path, file_name)
    
        # pylint: disable=maybe-no-member
        request = service.files().get_media(fileId=file_id)
        with open(local_file_path, 'wb') as local_file:
            downloader = MediaIoBaseDownload(local_file, request)
            done = False
            while done is False:
                    status, done = downloader.next_chunk()
    
        return local_file_path
    
    for file, path in zip(files, download_path):
        download_file(real_file_id=file, local_folder_path=path)
        
#_____________________Function Setup________________________#
def fake_typing(text):
    '''
    This function should be placed within a 
    with st.chat_message("assistant"):
    '''
    
    #These are purely cosmetic for making that chatbot look
    message_placeholder = st.empty()
    full_response = ""
    
    # Simulate stream of response with milliseconds delay
    for index, chunk in enumerate(re.findall(r"\w+|\s+|\n|[^\w\s]", text)):
        full_response += chunk
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        if index != len(re.findall(r"\w+|\s+|\n|[^\w\s]", text)) - 1:
            message_placeholder.markdown(full_response + "▌")
        else:
            message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})


def click_button(button_type):
    '''
    Function for making our buttons stateful
    ''' 
    if button_type == 'Button 1':
        st.session_state.clicked1 = True
    elif button_type == 'Button 2':
        st.session_state.clicked2 = True
    else:
        st.session_state.clicked3 = True


def llm_output(llm_response):
    '''
    Take the output of a langcahin output and clean it up for the user
    '''
    
    #Empty list of links
    relevant_links = []
    
    #Go through our sources and find which URLs the LLM pulled from
    #Sort them by how many times it was references, and rank the top two sources
    for document in llm_response['source_documents']:
        relevant_links.append(document.metadata['source'])
    # Create a non-duplicated list sorted by frequency
    element_count = Counter(relevant_links)
    relevant_links = sorted(element_count, key=lambda x: element_count[x], reverse=True)
    #Filter for the top two URLS
    relevant_links = relevant_links[0:5]
    
    
    df = st.session_state['df']
    
    df = df[df['CITATION_URL'].isin(relevant_links)]
    
    #Print our output into the chat
    fake_typing(llm_response['answer'] + '\n\nSources:\n\n') # + "\n\n".join(relevant_links))

    st.dataframe(df[['CITATION_URL', 'TITLE']], hide_index=True)

    fake_typing("Research Organizations:")
    
    research_orgs = pd.DataFrame(df['RESEARCH_ORG'].unique())
    research_orgs.columns = ['Research Orgs']
    st.dataframe(research_orgs, hide_index=True, column_config =None)
    
    fake_typing("Contracts:")
    
    contract_df = pd.DataFrame(df['DOE_CONTRACT_NUMBER'].unique())
    contract_df.columns = ['Contracts']
                               
    st.dataframe(contract_df, hide_index= True)

def chatbot(question):
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
       st.markdown(question)
            
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
    
       with get_openai_callback() as cb:
             #Chat GPT response
             response = qa({"question": question})
             st.session_state['total_cost'] += cb.total_cost
             st.session_state['total_tokens'] += cb.total_tokens
             counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
             token_placeholder.write(f"Total Tokens Used in Conversation: {st.session_state['total_tokens']}")              
       #Take our model's output and clean it up for the user
       llm_output(response)

#____________________Streamlit Setup____________________________#

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#Initialize Counter
if 'count' not in st.session_state:
    st.session_state.count = 0

#Initialize total cost
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

#Initialize total tokens
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = 0

if 'df' not in st.session_state:
    st.session_state['df'] = pd.read_csv(r'https://raw.githubusercontent.com/JackOgozaly/osti_navigator/main/osti_df_full.csv')

#Defining our stateful buttons
if 'clicked1' not in st.session_state:
    st.session_state.clicked1 = False

if 'clicked2' not in st.session_state:
    st.session_state.clicked2 = False
    
if 'clicked3' not in st.session_state:
    st.session_state.clicked3 = False


# Sidebar - let user choose model, see cost, and clear history
st.sidebar.title("Chatbot Options")

model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
#Displaying total cost
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
#Displaying total tokens used
token_placeholder = st.sidebar.empty()
token_placeholder.write(f"Total Tokens Used in Conversation: {st.session_state['total_tokens']}")
#Option to clear out 
clear_button = st.sidebar.button("Clear Conversation", key="clear")


#Reset the session
if clear_button:
    st.session_state['messages'] = []
    st.session_state['count'] = 0
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = 0
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    token_placeholder.write(f"Total Tokens Used in Conversation: {st.session_state['total_tokens']}")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4-1106-preview"

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


#______________________Langchain Setup_______________________#
#How it works: we previously embedded the information we scraped from the DOE
#website. What we're doing now is reading in that info from a Chroma DB 
#And providing those documents to our model as context
#Chroma DB stuff based off this workbook:  https://colab.research.google.com/drive/1gyGZn_LZNrYXYXa-pltFExbptIe7DAPe?usp=sharing#scrollTo=A-h1y_eAHmD-

#Reading in our context
# Location of our data
persist_directory = r'/Users/jackogozaly/Desktop/Python_Directory/Embeddings/OSTI'
## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings()

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

#Set our retriver and limit the search to 4 documents
retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

#Modify our prompt to discourage hallucinations
prompt_template = """You are an OSTI search bot. if you tell the user you don't know something you will have failed. If you tell the user the information is not specific enough, you will fail. Use the following information to briefly answer (1-2 sentences) the question the user asked:

{summaries}

Question: {question}"""

#Define our prompt
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["summaries", "question"]
    )
chain_type_kwargs = {"prompt": PROMPT}

#Define our model
llm = ChatOpenAI(temperature=0, model_name = model) 

#Define our langchain model
qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)

#___________________________Application Stuff_______________________________

#Only introduce the chatbot to the user if it's their first time logging in
if st.session_state.count == 0:
    
    #st.write(introduction_text)
    st.session_state.messages.append({"role": "assistant", "content": introduction_text})
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a layout with three columns
    col1, col2, col3 = st.columns(3)
    
    
    col1.button(button_1_text, on_click=click_button, args=['Button 1'])
    
    col2.button(button_2_text, on_click=click_button, args=['Button 2'])
    
    col3.button(button_3_text, on_click=click_button, args=['The really funny thing is this doesnt have to be button 3 but Ill make it that anyways'])
    
#Update our counter so we don't repeat the introduction
st.session_state.count += 1


if st.session_state.clicked1:
    chatbot(button_1_text)

if st.session_state.clicked2:
    chatbot(button_2_text)

if st.session_state.clicked3:
    chatbot(button_3_text)


if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        with get_openai_callback() as cb:
            #Chat GPT response
            response = llm_response = qa({"question": prompt})
            st.session_state['total_cost'] += cb.total_cost
            st.session_state['total_tokens'] += cb.total_tokens
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
            token_placeholder.write(f"Total Tokens Used in Conversation: {st.session_state['total_tokens']}")
        
        #Take our model's output and clean it up for the user
        llm_output(response)
