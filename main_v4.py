import streamlit as st
import google.generativeai as genai
import json
import time
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import base64
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

# Voice and Audio
from audiorecorder import audiorecorder # For STT
import openai # For Whisper STT
from gtts import gTTS # For TTS

# Markdown processing
import re
import markdown
from bs4 import BeautifulSoup

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Ribara Learning Platform",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Key Checks and OpenAI Client ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

if not GEMINI_API_KEY:
    st.error("🔴 GEMINI_API_KEY not found. Please set it in your environment variables or .env file.")
if not OPENAI_API_KEY:
    st.warning("🟡 OPENAI_API_KEY not found. Voice input (STT) for the AI Tutor will not function.")
else:
    openai.api_key = OPENAI_API_KEY

# --- Helper Functions ---
@st.cache_data(show_spinner=False)
def text_to_speech_audio(text: str, lang: str = 'en') -> BytesIO | None:
    cleaned_text = markdown_to_text(text)
    if not cleaned_text.strip(): return None
    try:
        tts = gTTS(text=cleaned_text, lang=lang, slow=False)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception:
        return None

def markdown_to_text(md_string: str) -> str:
    if not md_string: return ""
    try:
        html = markdown.markdown(md_string)
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=' ').strip()
    except Exception:
        return md_string

def extract_image_suggestions(text: str, tag_prefix: str = "IMAGE_SUGGESTION"):
    if not text: return "", []
    pattern = rf'\[{tag_prefix}:\s*(.*?)\]'
    queries = re.findall(pattern, text)
    cleaned_text = re.sub(pattern, '', text).strip()
    return cleaned_text, queries

@st.cache_data(show_spinner="Transcribing your speech...")
def transcribe_audio_with_whisper(_audio_bytes: bytes) -> str | None:
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not configured for transcription.")
        return None
    if not _audio_bytes:
        return None
    try:
        audio_file = BytesIO(_audio_bytes)
        audio_file.name = "temp_audio.wav" # Whisper API needs a named file or bytes
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript['text']
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

@st.cache_data(ttl=3600) # Cache images for an hour
def search_educational_images(query: str, num_images: int = 1) -> List[str]:
    try:
        if UNSPLASH_ACCESS_KEY:
            url = f"https://api.unsplash.com/search/photos"
            params = {'query': query, 'per_page': num_images, 'orientation': 'landscape'}
            headers = {'Authorization': f'Client-ID {UNSPLASH_ACCESS_KEY}'}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [photo['urls']['regular'] for photo in data['results']]
        
        st.sidebar.warning(f"Unsplash key missing or error. Using placeholder for '{query}'.")
        return [f"https://picsum.photos/600/300?random={hash(query)+i}&blur=1" for i in range(num_images)]
    except Exception as e:
        st.sidebar.warning(f"Image search error for '{query}': {e}")
        return [f"https://picsum.photos/600/300?random={hash(query)+i}&blur=1" for i in range(num_images)]

def display_images(images: List[str], caption_prefix: str = "Visual"):
    if not images: return
    cols = st.columns(min(len(images), 3))
    for idx, img_url in enumerate(images):
        with cols[idx % min(len(images), 3)]:
            try:
                st.image(img_url, caption=f"{caption_prefix} {idx+1}", use_container_width=True)
            except Exception:
                st.error("Could not load image.")

# --- LangChain Configuration ---
@st.cache_resource
def configure_langchain_llm_and_memory():
    if not GEMINI_API_KEY:
        return None, None
    genai.configure(api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", # Use a readily available model
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,
        convert_system_message_to_human=True
    )
    memory = ConversationBufferWindowMemory(k=8, return_messages=True, memory_key="chat_history")
    return llm, memory

class LearningPlatform:
    def __init__(self):
        self.llm, self.memory = configure_langchain_llm_and_memory()
        if self.llm and self.memory:
            self._setup_chains()
        else:
            st.error("LLM or Memory not initialized. AI features will be limited.")
            self.content_chain = None
            self.conversation_chain = None

    def _setup_chains(self):
        # Content Generation Chain (for main learning material)
        content_template_prompt = PromptTemplate(
            input_variables=["topic", "user_level", "previous_knowledge"],
            template="""
            You are an expert educational content creator. Create CONCISE learning content for:
            Topic: {topic}
            User Level: {user_level}
            Previous Knowledge/Context: {previous_knowledge}

            Provide:
            1. A clear, brief explanation suitable for {user_level}.
            2. 1-2 key practical examples or applications.
            3. A short summary of key takeaways.

            Format using Markdown. Keep it focused and not too lengthy.
            If a visual would significantly aid understanding for a specific point, suggest it like this:
            [IMAGE_SUGGESTION: concise query for an educational diagram or photo]
            """
        )
        self.content_chain = LLMChain(llm=self.llm, prompt=content_template_prompt)

        # Conversational Tutor Chain
        tutor_system_prompt = """You are an expert AI learning assistant specializing in engineering education.
You are interacting with the user via voice, so keep your responses concise and conversational, like in a spoken dialogue.
Provide information in smaller, digestible chunks (1-3 sentences typically) unless the user asks for more detail.
Your primary mode of response is speech; the text shown to the user is a caption of your speech.
Use Markdown for the text captions.

Always:
- Be clear, accurate, and encouraging.
- Use analogies or real-world examples if they are brief and very helpful.
- If the user asks for a practice problem, generate one.
- If the user asks for a short quiz, provide 1-2 multiple-choice questions.
- If you believe a visual would significantly clarify a point you're making, you can say something like, "A diagram of X might be helpful here." Then, in your text caption, include the tag [IMAGE_SEARCH: concise query for X diagram].
- Be patient and supportive. Ask clarifying questions if the user's query is vague.
"""
        tutor_prompt = ChatPromptTemplate.from_messages([
            ("system", tutor_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        self.conversation_chain = ConversationChain(
            llm=self.llm, memory=self.memory, prompt=tutor_prompt, verbose=False
        )

    def generate_learning_material(self, topic, user_level, prev_knowledge="None"):
        if not self.content_chain: return "Error: Content chain not ready."
        try:
            response = self.content_chain.run(
                topic=topic, user_level=user_level, previous_knowledge=prev_knowledge
            )
            return response
        except Exception as e:
            return f"Error generating learning material: {e}"

    def tutor_chat(self, user_input: str) -> str:
        if not self.conversation_chain: return "Error: Tutor chain not ready."
        try:
            # The memory is automatically updated by the ConversationChain
            response = self.conversation_chain.predict(input=user_input)
            return response
        except Exception as e:
            return f"Error in tutor response: {e}"

# --- Session State Initialization ---
def init_session_state():
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {'name': '', 'skill_level': 'Beginner'}
    if 'current_module_key' not in st.session_state:
        st.session_state.current_module_key = None
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = None
    if 'learning_material_cache' not in st.session_state: # For topic content
        st.session_state.learning_material_cache = {}

    # AI Tutor specific state
    if 'tutor_conversation' not in st.session_state: # List of {"speaker": "user/ai", "text": "...", "audio": BytesIO_or_None}
        st.session_state.tutor_conversation = []
    if 'ai_is_processing' not in st.session_state: # To show spinner and manage flow
        st.session_state.ai_is_processing = False
    if 'new_ai_audio' not in st.session_state: # To trigger autoplay
        st.session_state.new_ai_audio = None
    if 'audio_player_key' not in st.session_state:
        st.session_state.audio_player_key = str(time.time())


# --- Learning Modules Data --- (Simplified for this example)
LEARNING_MODULES = {
    'process_engineering': {
        'title': 'Process Engineering Essentials',
        'topics': ['Process Flow Diagrams', 'Mass and Energy Balance', 'Basic Heat Transfer']
    },
    'mechanical_engineering': {
        'title': 'Mechanical Engineering Fundamentals',
        'topics': ['Thermodynamics Basics', 'Intro to Mechanics of Materials', 'Simple Machine Design']
    },
    'robotics': {
        'title': 'Robotics Introduction',
        'topics': ['Robot Components', 'Basic Kinematics', 'Sensors in Robotics']
    }
}

# --- UI Rendering Functions ---
def render_sidebar(platform: LearningPlatform):
    st.sidebar.title("🗣️ AI Voice Learner")
    st.sidebar.markdown("---")

    # API Status
    if not GEMINI_API_KEY: st.sidebar.error("Gemini API Key Missing!")
    if not OPENAI_API_KEY: st.sidebar.warning("OpenAI API Key Missing (for STT)!")
    if not UNSPLASH_ACCESS_KEY: st.sidebar.caption("Unsplash Key missing (placeholders for images).")

    st.sidebar.subheader("👤 Profile")
    st.session_state.user_profile['name'] = st.sidebar.text_input(
        "Your Name", st.session_state.user_profile.get('name', ''), key="profile_name_sidebar"
    )
    st.session_state.user_profile['skill_level'] = st.sidebar.selectbox(
        "Skill Level", ['Beginner', 'Intermediate', 'Advanced'],
        index=['Beginner', 'Intermediate', 'Advanced'].index(st.session_state.user_profile.get('skill_level', 'Beginner')),
        key="profile_skill_sidebar"
    )
    if st.sidebar.button("Update Profile", key="update_profile_sidebar"):
        st.sidebar.success("Profile Updated!")
    st.sidebar.markdown("---")

    st.sidebar.subheader("📚 Learning Modules")
    if st.session_state.current_module_key and st.session_state.current_topic:
        if st.sidebar.button(f"← Back to All Modules", key="back_to_modules_btn"):
            st.session_state.current_module_key = None
            st.session_state.current_topic = None
            # Clear tutor conversation when changing modules/topics significantly
            if platform.memory: platform.memory.clear()
            st.session_state.tutor_conversation = []
            st.rerun()

    selected_module_key = st.sidebar.radio(
        "Choose a Module:", list(LEARNING_MODULES.keys()),
        format_func=lambda key: LEARNING_MODULES[key]['title'],
        key="module_selector",
        index=list(LEARNING_MODULES.keys()).index(st.session_state.current_module_key) if st.session_state.current_module_key else 0
    )

    if selected_module_key != st.session_state.current_module_key:
        st.session_state.current_module_key = selected_module_key
        st.session_state.current_topic = None # Reset topic when module changes
        if platform.memory: platform.memory.clear()
        st.session_state.tutor_conversation = []
        st.rerun()

    if st.session_state.current_module_key:
        module = LEARNING_MODULES[st.session_state.current_module_key]
        st.sidebar.markdown(f"**Topics in {module['title']}:**")
        
        # Use buttons for topic selection for clearer action
        for topic_item in module['topics']:
            is_current_topic = (st.session_state.current_topic == topic_item)
            button_type = "primary" if is_current_topic else "secondary"
            if st.sidebar.button(topic_item, key=f"topic_btn_{topic_item.replace(' ','_')}", type=button_type, use_container_width=True):
                if st.session_state.current_topic != topic_item:
                    st.session_state.current_topic = topic_item
                    # Clear conversation when topic changes
                    if platform.memory: platform.memory.clear()
                    st.session_state.tutor_conversation = []
                    # Clear cached material for previous topic to force regeneration if needed
                    # Or, just let it regenerate when "View/Refresh Learning Material" is clicked
                st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.info("Tip: Ask the AI Tutor for practice, quizzes, or to explain concepts from the material!")


def render_learning_space(platform: LearningPlatform):
    if not st.session_state.current_module_key or not st.session_state.current_topic:
        st.info("👋 Welcome! Please select a module and topic from the sidebar to begin.")
        if st.session_state.user_profile.get('name'):
            st.write(f"Hello, {st.session_state.user_profile['name']}!")
        return

    module = LEARNING_MODULES[st.session_state.current_module_key]
    topic = st.session_state.current_topic
    st.header(f"📖 {module['title']}")
    st.subheader(f"Topic: {topic}")
    st.markdown("---")

    # Learning Material Section
    material_cache_key = f"{st.session_state.current_module_key}_{topic}_material"
    
    if st.button("🔄 View / Refresh Learning Material", key="refresh_material_btn"):
        with st.spinner(f"Loading material for {topic}..."):
            prev_topics_context = ", ".join(list(st.session_state.learning_material_cache.keys())) # Simple context
            material_text = platform.generate_learning_material(
                topic, st.session_state.user_profile['skill_level'], prev_topics_context
            )
            st.session_state.learning_material_cache[material_cache_key] = material_text
            st.rerun() # Rerun to display the new material

    if material_cache_key in st.session_state.learning_material_cache:
        with st.container(border=True):
            st.markdown("#### 🧠 Learning Material")
            material_content = st.session_state.learning_material_cache[material_cache_key]
            
            cleaned_material, image_suggestions = extract_image_suggestions(material_content, "IMAGE_SUGGESTION")
            st.markdown(cleaned_material)

            if image_suggestions:
                st.markdown("---")
                st.markdown("##### Visual Aids Suggested by Material:")
                for i, query in enumerate(image_suggestions):
                    with st.spinner(f"Fetching image for: {query}"):
                        imgs = search_educational_images(query, 1)
                    display_images(imgs, caption_prefix=f"Suggested: {query[:30]}...")
            
            # TTS for main learning material
            if st.button(f"🔊 Read Material Aloud", key=f"tts_main_material_{topic.replace(' ','_')}"):
                with st.spinner("Generating audio for material..."):
                    audio_bytes = text_to_speech_audio(cleaned_material)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.warning("Could not generate audio for the material.")
    else:
        st.info("Click 'View / Refresh Learning Material' to load the content for this topic.")

    st.markdown("---")
    render_ai_tutor(platform, topic)


def render_ai_tutor(platform: LearningPlatform, current_topic: str):
    st.subheader("🎙️ AI Tutor")
    st.caption(f"Interacting about: **{current_topic}**. Speak your questions or requests.")

    # Display conversation history
    if st.session_state.tutor_conversation:
        for entry in st.session_state.tutor_conversation:
            with st.chat_message(entry["speaker"]):
                st.markdown(entry["text"])
                # If AI, and it had image suggestions, display them
                if entry["speaker"] == "ai":
                    _, image_queries = extract_image_suggestions(entry["text"], "IMAGE_SEARCH")
                    if image_queries:
                        for i, query in enumerate(image_queries):
                            with st.spinner(f"Loading image for: {query}"):
                                imgs = search_educational_images(query, 1)
                            display_images(imgs, caption_prefix=f"AI Visual: {query[:30]}...")
    else:
        st.info("No conversation yet. Tap the button below to speak to the AI Tutor!")


    # Audio recorder and processing logic
    # Put audiorecorder in a column to control its width if needed
    recorder_col, _, _ = st.columns([2,1,1])
    with recorder_col:
        _audio_bytes = audiorecorder(
            "Tap to Speak", "Listening... (Tap again to Stop)", 
            key="audiorec")
    
    if _audio_bytes and not st.session_state.ai_is_processing:
        st.session_state.ai_is_processing = True # Lock to prevent multiple processing
        
        # 1. User Speech to Text
        user_transcription = transcribe_audio_with_whisper(_audio_bytes)
        
        if user_transcription:
            st.session_state.tutor_conversation.append({"speaker": "user", "text": user_transcription})
            
            # 2. Get AI Response from LangChain
            # The context (current_topic) is implicitly part of the conversation history now,
            # or could be added to the user_input string if needed for the first turn on a new topic.
            # For this turn, let's ensure the topic is passed to the tutor.
            ai_response_text = platform.tutor_chat(f"Regarding {current_topic}: {user_transcription}")
            
            # 3. AI Text to Speech
            ai_audio_bytes = text_to_speech_audio(ai_response_text)
            
            st.session_state.tutor_conversation.append({"speaker": "ai", "text": ai_response_text, "audio_bytes": ai_audio_bytes})
            
            if ai_audio_bytes:
                st.session_state.new_ai_audio = ai_audio_bytes # Signal to play this audio
                st.session_state.audio_player_key = f"audio_{time.time()}" # Unique key for autoplay

        else:
            st.warning("Could not understand your speech. Please try again.")
        
        st.session_state.ai_is_processing = False # Unlock
        st.rerun() # Rerun to display new messages and play audio

    # Autoplay AI's new audio response
    if st.session_state.new_ai_audio:
        st.audio(st.session_state.new_ai_audio, format="audio/mp3", autoplay=True, key=st.session_state.audio_player_key)
        st.session_state.new_ai_audio = None # Clear after attempting to play

    if st.session_state.ai_is_processing:
        st.info("AI is thinking...")


# --- Main Application ---
def main():
    init_session_state()
    platform = LearningPlatform() # Initialize once

    render_sidebar(platform)
    
    # Main content area
    main_col, _ = st.columns([3,1]) # Main content a bit wider
    with main_col:
        # Based on sidebar navigation, render the main content
        # For now, we only have the "learning space" which is module/topic driven
        render_learning_space(platform)

    # Removed separate pages for Dashboard, Analytics, Settings for extreme simplification as requested.
    # These could be added back if essential, possibly as modal dialogs or simpler sections.
    # Profile is handled in the sidebar.

if __name__ == "__main__":
    main()