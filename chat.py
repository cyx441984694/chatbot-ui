
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

@st.cache_resource(show_spinner=False)
def LLM_init():
    template = """
    {chat_history}
        Question: {human_input}
        Answer: """

    promptllm = PromptTemplate(template=template, input_variables=["chat_history","human_input"])
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    llm_chain = LLMChain(
        prompt=promptllm, 
        llm=VertexAI(), 
        memory=memory, 
        verbose=True
    )
    
    return llm_chain


st.set_page_config(page_title="Chatbot App")
st.title('🦜🔗 Chatbot App')

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi my name is CMLP and I am your assistant, how can I help you?"}]

#"st.session_state:", st.session_state.messages

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # with st.spinner('Preparing'):
    llm_chain = LLM_init()
    msg = llm_chain.run(human_input=prompt)

    #st.write(msg)

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)