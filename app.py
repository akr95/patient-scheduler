import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def safe_json_parse(input_str):
    try:
        return json.loads(input_str.replace("'", '"'))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")

def check_availability(date, time):
    return f"✅ Available on {date} at {time}."

def book_appointment(name, date, time):
    return f"📅 Appointment booked for {name} on {date} at {time}."

tools = [
    Tool(
        name="CheckAvailability",
        func=lambda x: check_availability(**safe_json_parse(x)),
        description="Check if a doctor is available. Input: JSON string with 'date' and 'time'."
    ),
    Tool(
        name="BookAppointment",
        func=lambda x: book_appointment(**safe_json_parse(x)),
        description="Book an appointment. Input: JSON string with 'name', 'date', and 'time'."
    )
]

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

st.title("🦷 Patient Scheduling Assistant")
name = st.text_input("Patient Name", "John")
date = st.date_input("Appointment Date")
time = st.time_input("Appointment Time")
submit = st.button("Schedule Appointment")

if submit:
    query = f"Can you schedule a teeth cleaning for {name} on {date.strftime('%B %d')} at {time.strftime('%I:%M %p')}?"
    with st.spinner("Talking to the agent..."):
        response = agent_executor.run(query)
    st.success("✅ Appointment scheduled!")
    st.write(response)
