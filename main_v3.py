import streamlit as st
import google.generativeai as genai
import json
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import base64
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.sequential import SequentialChain

# New imports for TTS and Markdown processing
from gtts import gTTS
import re
import markdown
from bs4 import BeautifulSoup

load_dotenv()

# Configure page
st.set_page_config(
    page_title="Ribara Learning Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions for TTS and Image Suggestions ---
@st.cache_data(show_spinner=False) # Cache audio generation for the same text
def text_to_speech_audio(text: str, lang: str = 'en'):
    """Converts text to an MP3 audio stream in memory using gTTS."""
    try:
        cleaned_text = markdown_to_text(text) # Ensure plain text for TTS
        if not cleaned_text.strip():
            return None # Don't generate audio for empty strings
        tts = gTTS(text=cleaned_text, lang=lang, slow=False)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except AssertionError: # Handles empty string issue with gTTS
        st.warning("Cannot generate audio for empty or whitespace-only text.")
        return None
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

def markdown_to_text(md_string: str) -> str:
    """Converts a markdown string to plain text."""
    if not md_string:
        return ""
    try:
        html = markdown.markdown(md_string)
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=' ').strip()
    except Exception:
        return md_string # Fallback to original string if parsing fails

def extract_image_suggestions(text: str, tag_prefix: str = "IMAGE_SUGGESTION"):
    """
    Extracts image suggestion queries from text and returns cleaned text and queries.
    Example tag: [IMAGE_SUGGESTION: query for image search here]
    """
    if not text:
        return "", []
    pattern = rf'\[{tag_prefix}:\s*(.*?)\]'
    queries = re.findall(pattern, text)
    cleaned_text = re.sub(pattern, '', text).strip()
    return cleaned_text, queries

# --- End Helper Functions ---


# Configure Gemini API and LangChain
@st.cache_resource
def configure_langchain_llm():
    """Configure LangChain with Google Gemini"""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Please set GEMINI_API_KEY in environment variables")
        return None, None
    
    genai.configure(api_key=api_key)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", # Changed to a generally available model, adjust if needed
        google_api_key=api_key,
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    memory = ConversationBufferWindowMemory(
        k=10,
        return_messages=True,
        memory_key="chat_history"
    )
    
    return llm, memory

@st.cache_data
def search_educational_images(query: str, num_images: int = 3) -> List[str]:
    """Search for educational images using Unsplash API"""
    try:
        access_key = os.getenv("UNSPLASH_ACCESS_KEY", "")
        
        if access_key:
            url = f"https://api.unsplash.com/search/photos"
            params = {
                'query': f"{query} engineering education diagram", # Appends context
                'per_page': num_images,
                'orientation': 'landscape'
            }
            headers = {'Authorization': f'Client-ID {access_key}'}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return [photo['urls']['regular'] for photo in data['results']]
        
        # Fallback to placeholder images if no API key or error
        st.warning(f"Unsplash API key not found or error. Using placeholder images for '{query}'.")
        return [f"https://picsum.photos/800/400?random={hash(query)+i}&blur=1" for i in range(num_images)]
        
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch images due to network issue: {str(e)}")
        return [f"https://picsum.photos/800/400?random={hash(query)+i}&blur=1" for i in range(num_images)]
    except Exception as e:
        st.warning(f"Could not fetch images: {str(e)}")
        return [f"https://picsum.photos/800/400?random={hash(query)+i}&blur=1" for i in range(num_images)]


def display_images_with_context(images: List[str], context: str, num_cols: int = 3):
    """Display images with educational context"""
    if not images:
        return
    
    # st.subheader("📸 Visual Learning Aids") # Title can be contextual
    
    cols = st.columns(min(len(images), num_cols))
    for idx, img_url in enumerate(images[:num_cols]):
        with cols[idx % num_cols]:
            try:
                response = requests.get(img_url, timeout=10)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption=f"Illustration for: {context}" if context else f"Illustration {idx+1}", use_container_width=True)
            except Exception as e:
                st.error(f"Could not load image {idx+1}: {e}")

# Initialize session state
def init_session_state():
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'skill_level': 'Beginner',
            'learning_goals': [],
            'completed_assessments': [],
            'progress_data': {},
            'chat_history': [] # This might be superseded by LangChain memory for actual chat logic
        }
    
    if 'current_module' not in st.session_state:
        st.session_state.current_module = None
    
    if 'assessment_results' not in st.session_state:
        st.session_state.assessment_results = {}
    
    if 'langchain_memory' not in st.session_state:
        _, memory = configure_langchain_llm()
        st.session_state.langchain_memory = memory # Storing the actual memory object
    
    if 'learning_context' not in st.session_state:
        st.session_state.learning_context = {}
    
    # For storing generated content and image queries per topic
    if 'generated_content_cache' not in st.session_state:
        st.session_state.generated_content_cache = {}

# Learning modules data structure (unchanged)
LEARNING_MODULES = {
    'process_engineering': {
        'title': 'Process Engineering',
        'description': 'Fundamentals of process design and optimization',
        'levels': ['Beginner', 'Intermediate', 'Advanced'],
        'topics': [
            'Process Flow Diagrams',
            'Mass and Energy Balance',
            'Heat Transfer',
            'Fluid Mechanics',
            'Reactor Design'
        ]
    },
    'mechanical_engineering': {
        'title': 'Mechanical Engineering',
        'description': 'Core mechanical engineering principles',
        'levels': ['Beginner', 'Intermediate', 'Advanced'],
        'topics': [
            'Thermodynamics',
            'Mechanics of Materials',
            'Machine Design',
            'Manufacturing Processes',
            'CAD/CAM'
        ]
    },
    'robotics': {
        'title': 'Robotics',
        'description': 'Introduction to robotics and automation',
        'levels': ['Beginner', 'Intermediate', 'Advanced'],
        'topics': [
            'Robot Kinematics',
            'Control Systems',
            'Sensors and Actuators',
            'Programming',
            'AI in Robotics'
        ]
    }
}

class EnhancedLearningPlatform:
    def __init__(self):
        self.llm, self.memory = configure_langchain_llm()
        if not self.llm: # Ensure llm is configured before setting up chains
            st.error("LLM not configured. AI features will be unavailable.")
            self.content_chain = None
            self.assessment_chain = None
            self.feedback_chain = None
            self.conversation_chain = None
            return
        self.setup_chains()
    
    def setup_chains(self):
        if not self.llm:
            return
        
        content_template = PromptTemplate(
            input_variables=["topic", "user_level", "learning_style", "previous_knowledge"],
            template="""
            You are an expert educational content creator. Create personalized learning content for:
            
            Topic: {topic}
            User Level: {user_level}
            Learning Style: {learning_style}
            Previous Knowledge: {previous_knowledge}
            
            Please provide:
            1. A clear explanation suitable for {user_level} level, using Markdown for formatting.
            2. 3 practical real-world examples or applications.
            3. 2 hands-on exercises with step-by-step guidance.
            4. Key takeaways and summary.
            5. Connection to previous topics learned.
            
            Format the response with clear sections (e.g., using ### Headings) and use analogies where appropriate.
            Make it engaging and interactive.
            Where relevant, suggest appropriate search queries for educational images that would enhance understanding. 
            Format these suggestions as: [IMAGE_SUGGESTION: query for image search here]. For example: [IMAGE_SUGGESTION: diagram of a heat exchanger].
            Ensure suggestions are distinct and appear where they are most relevant in the text.
            """
        )
        
        assessment_template = PromptTemplate(
            input_variables=["topic", "level", "learning_context"],
            template="""
            Create a comprehensive assessment for {topic} at {level} level.
            Learning Context: {learning_context}
            
            Generate a JSON response with:
            1. 5 multiple choice questions with explanations
            2. 2 practical scenarios requiring problem-solving
            3. 1 design challenge with clear criteria
            
            Ensure questions build upon each other and test understanding at different cognitive levels.
            
            JSON Format:
            {{
                "multiple_choice": [
                    {{"question": "Question text", "options": ["A", "B", "C", "D"], "correct": 0, "explanation": "Detailed explanation", "difficulty": "easy/medium/hard"}}
                ],
                "scenarios": [
                    {{"problem": "Real-world scenario description", "context": "Background information", "solution_approach": "Expected approach", "key_concepts": ["concept1", "concept2"]}}
                ],
                "design_challenge": {{"challenge": "Design task description", "constraints": ["constraint1", "constraint2"], "criteria": ["criterion1", "criterion2"], "deliverables": ["deliverable1", "deliverable2"]}}
            }}
            """
        )
        
        feedback_template = PromptTemplate(
            input_variables=["question", "user_answer", "correct_answer", "user_context"],
            template="""
            Evaluate this learning response with personalized feedback:
            
            Question: {question}
            User Answer: {user_answer}
            Expected Answer: {correct_answer}
            User Context: {user_context}
            
            Provide in Markdown format:
            1. Score (0-100) with justification
            2. Detailed constructive feedback
            3. Specific areas for improvement
            4. Personalized recommendations for next steps
            5. Encouragement and motivation
            6. Additional resources or practice suggestions
            
            Be encouraging while being specific about improvements needed.
            """
        )
        
        self.content_chain = LLMChain(llm=self.llm, prompt=content_template, output_key="content")
        self.assessment_chain = LLMChain(llm=self.llm, prompt=assessment_template, output_key="assessment")
        self.feedback_chain = LLMChain(llm=self.llm, prompt=feedback_template, output_key="feedback")
        
        conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI learning assistant specializing in engineering education. 
            You have access to the conversation history and can reference previous topics discussed.
            
            Always:
            - Provide clear, accurate technical explanations using Markdown for good readability.
            - Use analogies and real-world examples.
            - Encourage hands-on learning.
            - Reference previous learning when relevant.
            - Suggest visual aids and diagrams when helpful by providing a search query like '[IMAGE_SEARCH: your query for an image]'. For example: '[IMAGE_SEARCH: types of robotic grippers]'. Ensure these tags are on their own line or clearly separable.
            - Be patient and supportive.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory, # Use the shared memory instance
            prompt=conversation_prompt,
            verbose=True # Set to False for production to reduce console noise
        )
    
    def generate_personalized_content(self, topic, user_level, learning_style, context=""):
        if not self.content_chain: return "Error: Content generation chain not initialized."
        try:
            previous_knowledge = st.session_state.learning_context.get('previous_topics', 'None')
            response = self.content_chain.run(
                topic=topic,
                user_level=user_level,
                learning_style=learning_style,
                previous_knowledge=previous_knowledge
            )
            if topic not in st.session_state.learning_context.get('previous_topics', ''):
                if 'previous_topics' not in st.session_state.learning_context or not st.session_state.learning_context['previous_topics']:
                    st.session_state.learning_context['previous_topics'] = topic
                else:
                    st.session_state.learning_context['previous_topics'] += f", {topic}"
            return response
        except Exception as e:
            return f"Error generating content: {str(e)}"
    
    def create_assessment(self, topic, level):
        if not self.assessment_chain: return {"error": "Assessment chain not initialized."}
        try:
            learning_context_str = st.session_state.learning_context.get('previous_topics', 'None relevant')
            response_str = self.assessment_chain.run(
                topic=topic,
                level=level,
                learning_context=learning_context_str
            )
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', response_str) # Look for markdown code block or raw JSON
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                return json.loads(json_str)
            else: # Fallback for direct JSON output without markdown
                json_start = response_str.find('{')
                json_end = response_str.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = response_str[json_start:json_end]
                    return json.loads(json_str)
                return {"error": "Could not parse assessment JSON. LLM did not return valid JSON."}
        except json.JSONDecodeError as e:
            return {"error": f"Error decoding assessment JSON: {str(e)}. Raw response: {response_str[:500]}..."}
        except Exception as e:
            return {"error": f"Error creating assessment: {str(e)}"}

    def evaluate_response(self, question, user_answer, correct_answer):
        if not self.feedback_chain: return "Error: Feedback chain not initialized."
        try:
            user_context_dict = {
                'skill_level': st.session_state.user_profile['skill_level'],
                'learning_goals': st.session_state.user_profile.get('learning_goals', []),
                'previous_topics': st.session_state.learning_context.get('previous_topics', 'None')
            }
            response = self.feedback_chain.run(
                question=question,
                user_answer=user_answer,
                correct_answer=correct_answer,
                user_context=str(user_context_dict)
            )
            return response
        except Exception as e:
            return f"Error evaluating response: {str(e)}"
    
    def chat_with_memory(self, user_input, topic_context=""):
        if not self.conversation_chain: return "Error: Conversation chain not initialized."
        try:
            # LangChain memory is updated automatically by the ConversationChain
            # The memory object is self.memory
            enhanced_input = user_input
            if topic_context: # Add context if provided
                enhanced_input = f"Regarding the topic of {topic_context}: {user_input}"
            
            response = self.conversation_chain.predict(input=enhanced_input)
            return response
        except Exception as e:
            st.error(f"Chat error: {e}")
            return f"Error in chat response: {str(e)}"

# --- Streamlit UI Rendering Functions ---

def render_user_profile():
    st.header("👤 User Profile")
    with st.form("profile_form"):
        name = st.text_input("Name", value=st.session_state.user_profile['name'])
        skill_level = st.selectbox(
            "Overall Skill Level",
            ['Beginner', 'Intermediate', 'Advanced'],
            index=['Beginner', 'Intermediate', 'Advanced'].index(st.session_state.user_profile['skill_level'])
        )
        learning_goals = st.multiselect(
            "Learning Goals",
            ['Career Advancement', 'Academic Study', 'Personal Interest', 'Certification'],
            default=st.session_state.user_profile['learning_goals']
        )
        learning_style = st.selectbox( # Added learning_style to profile
            "Preferred Learning Style",
            ['Visual', 'Auditory', 'Kinesthetic', 'Reading/Writing'],
            index=['Visual', 'Auditory', 'Kinesthetic', 'Reading/Writing'].index(st.session_state.user_profile.get('learning_style', 'Visual'))
        )
        submitted = st.form_submit_button("Update Profile")
        if submitted:
            st.session_state.user_profile.update({
                'name': name, 'skill_level': skill_level, 'learning_goals': learning_goals, 'learning_style': learning_style
            })
            st.success("Profile updated successfully!")

def render_learning_dashboard():
    st.header("📚 Learning Dashboard")
    if st.session_state.user_profile['name']:
        st.write(f"Welcome back, **{st.session_state.user_profile['name']}**! 👋")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Modules Started", len(st.session_state.learning_context.get('previous_topics', '').split(',')) if st.session_state.learning_context.get('previous_topics') else 0)
    with col2: 
        total_topics_all_modules = sum(len(m['topics']) for m in LEARNING_MODULES.values())
        st.metric("Total Topics Available", total_topics_all_modules)
    with col3: st.metric("Skill Level", st.session_state.user_profile['skill_level'])
    with col4: st.metric("Assessments Taken", len(st.session_state.assessment_results))

    if st.session_state.learning_context.get('previous_topics'):
        st.info(f"🎯 **Learning Context**: You've been exploring {st.session_state.learning_context['previous_topics']}")
    
    st.subheader("Available Learning Modules")
    module_keys = list(LEARNING_MODULES.keys())
    cols = st.columns(len(module_keys))
    for idx, key in enumerate(module_keys):
        module = LEARNING_MODULES[key]
        with cols[idx]:
            with st.container(border=True):
                st.markdown(f"**{module['title']}**")
                st.caption(module['description'])
                st.write(f"Topics: {len(module['topics'])}")
                if st.button(f"Explore {module['title']}", key=f"start_{key}", type="primary", use_container_width=True):
                    st.session_state.current_module = key
                    st.rerun()

def render_learning_module(module_key):
    module = LEARNING_MODULES[module_key]
    platform = EnhancedLearningPlatform() # Initialize platform with chains
    
    st.header(f"📖 {module['title']}")
    if st.button("← Back to Dashboard"):
        st.session_state.current_module = None
        st.rerun()

    selected_topic = st.selectbox("Select Topic", module['topics'], key=f"topic_select_{module_key}")
    
    # Cache key for generated content for the current topic
    content_cache_key = f"{module_key}_{selected_topic}_content"
    image_queries_cache_key = f"{module_key}_{selected_topic}_image_queries"

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Learn", "🖼️ Visual Search", "🧪 Practice", "📊 Assessment", "💬 AI Tutor"])

    with tab1:
        st.subheader("Personalized Learning Content")
        
        # Use profile's learning style or allow override
        default_style_index = ['Visual', 'Auditory', 'Kinesthetic', 'Reading/Writing'].index(st.session_state.user_profile.get('learning_style', 'Visual'))
        learning_style_override = st.radio(
            "Adjust Learning Approach for this Topic",
            ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"],
            index=default_style_index,
            horizontal=True
        )
        
        if st.button("Generate Personalized Content", type="primary", key=f"gen_content_{selected_topic}"):
            with st.spinner("Creating personalized learning experience..."):
                if platform.llm: # Check if LLM is available
                    raw_content = platform.generate_personalized_content(
                        selected_topic,
                        st.session_state.user_profile['skill_level'],
                        learning_style_override # Use the override
                    )
                    cleaned_content, image_queries = extract_image_suggestions(raw_content, "IMAGE_SUGGESTION")
                    st.session_state.generated_content_cache[content_cache_key] = cleaned_content
                    st.session_state.generated_content_cache[image_queries_cache_key] = image_queries
                else:
                    st.error("LLM not available. Cannot generate content.")
                    st.session_state.generated_content_cache[content_cache_key] = "Error: LLM not available."
                    st.session_state.generated_content_cache[image_queries_cache_key] = []
                st.rerun() # Rerun to display generated content and TTS button correctly

        # Display generated content if available
        if content_cache_key in st.session_state.generated_content_cache:
            content_to_display = st.session_state.generated_content_cache[content_cache_key]
            st.markdown("--- \n ### 📚 Learning Material")
            st.markdown(content_to_display)

            # TTS Button for generated content
            if content_to_display and "Error:" not in content_to_display:
                if st.button("🔊 Read Content Aloud", key=f"tts_content_{selected_topic}"):
                    with st.spinner("Generating audio..."):
                        audio_bytes = text_to_speech_audio(content_to_display)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                    else:
                        st.warning("Could not generate audio for the content.")
            
            # Display suggested images from content
            suggested_queries = st.session_state.generated_content_cache.get(image_queries_cache_key, [])
            if suggested_queries:
                st.markdown("--- \n ### 🖼️ Suggested Visuals from Content")
                for query_idx, img_query in enumerate(suggested_queries):
                    with st.container(border=True):
                        st.caption(f"AI suggested visual for: *{img_query}*")
                        with st.spinner(f"Searching for '{img_query}'..."):
                            images = search_educational_images(img_query, num_images=1)
                        if images:
                            display_images_with_context(images, img_query, num_cols=1)
                        else:
                            st.caption(f"_No images found for: {img_query}_")
        else:
            st.info("Click 'Generate Personalized Content' to load learning materials.")


    with tab2: # Visual Search Tab (User-initiated search)
        st.subheader("🖼️ Visual Learning Aids Search")
        search_query_visuals = st.text_input("Search for educational images (e.g., 'thermodynamics chart')", f"{selected_topic} diagram")
        
        if st.button("Find Images", key=f"find_images_{selected_topic}"):
            with st.spinner(f"Searching for images related to '{search_query_visuals}'..."):
                images = search_educational_images(search_query_visuals, num_images=3)
                st.session_state[f"user_searched_images_{selected_topic}"] = images

        if f"user_searched_images_{selected_topic}" in st.session_state:
            images_to_display = st.session_state[f"user_searched_images_{selected_topic}"]
            if images_to_display:
                display_images_with_context(images_to_display, search_query_visuals)
            else:
                st.warning("No images found for your search query.")

    with tab3: # Practice Tab
        st.subheader("🧪 Interactive Practice")
        st.markdown("### Hands-on Learning Exercise")
        exercise_type = st.selectbox("Choose Exercise Type", ["Problem Solving", "Design Challenge", "Case Study", "Calculation Practice"], key=f"ex_type_{selected_topic}")

        if st.button("Generate Practice Exercise", key=f"gen_ex_{selected_topic}"):
            with st.spinner("Creating personalized exercise..."):
                if platform.llm:
                    exercise_prompt = f"Create a {exercise_type.lower()} exercise for {selected_topic} at {st.session_state.user_profile['skill_level']} level. Include: 1. Clear problem statement. 2. Required background knowledge. 3. Step-by-step approach hints. 4. Expected deliverables. Format using Markdown."
                    exercise = platform.chat_with_memory(exercise_prompt, selected_topic) # Use chat for this type of generation
                    st.session_state[f"exercise_{selected_topic}"] = exercise
                else:
                    st.error("LLM not available.")
        
        if f"exercise_{selected_topic}" in st.session_state:
            st.markdown("--- \n ### 📋 Your Exercise")
            st.markdown(st.session_state[f"exercise_{selected_topic}"])
            
            exercise_answer = st.text_area("Your solution/approach:", height=150, key=f"ex_ans_{selected_topic}", placeholder="Explain your approach...")
            if st.button("Submit & Get Feedback", key=f"submit_ex_{selected_topic}") and exercise_answer:
                with st.spinner("Analyzing your solution..."):
                    if platform.llm:
                        feedback = platform.evaluate_response(
                            f"{exercise_type} for {selected_topic}: {st.session_state[f'exercise_{selected_topic}'][:200]}...", # Question context
                            exercise_answer,
                            "A comprehensive solution demonstrating understanding and proper methodology." # Expected answer context
                        )
                        st.markdown("--- \n ### 📝 Personalized Feedback")
                        st.markdown(feedback)
                    else:
                        st.error("LLM not available for feedback.")
    
    with tab4: # Assessment Tab
        st.subheader("📊 Adaptive Assessment")
        difficulty = st.selectbox("Assessment Difficulty", ["Basic Understanding", "Applied Knowledge", "Advanced Problem Solving"], key=f"assess_diff_{selected_topic}")

        if st.button("Generate Smart Assessment", key=f"gen_assess_{selected_topic}"):
            with st.spinner("Creating adaptive assessment..."):
                if platform.llm:
                    assessment_data = platform.create_assessment(selected_topic, difficulty)
                    if 'error' not in assessment_data:
                        st.session_state.current_assessment_data = assessment_data # Store data
                        st.session_state.current_assessment_topic = selected_topic
                    else:
                        st.error(assessment_data['error'])
                        st.session_state.current_assessment_data = None
                else:
                    st.error("LLM not available for assessment generation.")
                st.rerun() # Rerun to display assessment

        if 'current_assessment_data' in st.session_state and st.session_state.current_assessment_data and st.session_state.current_assessment_topic == selected_topic:
            render_enhanced_assessment(st.session_state.current_assessment_data, selected_topic, platform)
        elif st.session_state.get('current_assessment_topic') == selected_topic and not st.session_state.get('current_assessment_data'):
             st.warning("Assessment could not be generated. Please try again or check API settings.")


    with tab5: # AI Tutor Tab
        render_enhanced_chat_interface(selected_topic, platform)


def render_enhanced_assessment(assessment_data, topic, platform):
    st.write(f"### 🎯 Adaptive Assessment for {topic}")
    user_answers = st.session_state.get(f"user_answers_{topic}", {})

    with st.form(key=f"assessment_form_{topic}"):
        if 'multiple_choice' in assessment_data:
            st.markdown("#### Multiple Choice Questions")
            for i, q in enumerate(assessment_data['multiple_choice']):
                st.markdown(f"**{i+1}. {q['question']}** (Difficulty: {q.get('difficulty', 'N/A')})")
                options = q['options']
                # Ensure unique key for radio button within the form for this attempt
                user_answers[f"mc_{i}"] = st.radio("Select answer:", options, key=f"mc_{topic}_{i}", index=None)
        
        if 'scenarios' in assessment_data:
            st.markdown("#### Real-World Scenarios")
            for i, scenario in enumerate(assessment_data['scenarios']):
                st.markdown(f"**Scenario {i+1}: {scenario['problem']}**")
                st.caption(f"Context: {scenario.get('context', 'N/A')}")
                st.caption(f"Key Concepts: {', '.join(scenario.get('key_concepts', []))}")
                user_answers[f"scenario_{i}"] = st.text_area("Your detailed approach:", key=f"scenario_{topic}_{i}", height=120)

        if 'design_challenge' in assessment_data:
            st.markdown("#### Design Challenge")
            challenge = assessment_data['design_challenge']
            st.markdown(f"**{challenge['challenge']}**")
            st.caption(f"Constraints: {', '.join(challenge.get('constraints', []))}")
            st.caption(f"Criteria: {', '.join(challenge.get('criteria', []))}")
            user_answers[f"design_challenge"] = st.text_area("Your design solution:", key=f"design_{topic}", height=150)

        submitted = st.form_submit_button("Submit Assessment")

    if submitted:
        st.session_state[f"user_answers_{topic}"] = user_answers # Save submitted answers
        st.session_state.assessment_results[topic] = {'submitted_answers': user_answers, 'assessment_structure': assessment_data}
        
        # Process and display results & feedback (simplified for brevity, expand as needed)
        st.markdown("--- \n ### 📊 Assessment Results & Feedback")
        mc_correct_count = 0
        mc_total_count = 0
        if 'multiple_choice' in assessment_data:
            mc_total_count = len(assessment_data['multiple_choice'])
            for i, q in enumerate(assessment_data['multiple_choice']):
                user_ans = user_answers.get(f"mc_{i}")
                correct_ans = q['options'][q['correct']]
                if user_ans == correct_ans:
                    mc_correct_count += 1
                    st.success(f"MCQ {i+1}: Correct! Your answer: {user_ans}. Explanation: {q.get('explanation', 'N/A')}")
                else:
                    st.error(f"MCQ {i+1}: Incorrect. Your answer: {user_ans}. Correct: {correct_ans}. Explanation: {q.get('explanation', 'N/A')}")
            if mc_total_count > 0:
                st.metric("Multiple Choice Score", f"{(mc_correct_count/mc_total_count)*100:.1f}% ({mc_correct_count}/{mc_total_count})")

        # Feedback for open-ended questions
        for key, data in user_answers.items():
            if (key.startswith('scenario_') or key == 'design_challenge') and data: # If answer provided
                question_text = ""
                expected_approach = ""
                if key.startswith('scenario_'):
                    q_idx = int(key.split('_')[1])
                    question_text = assessment_data['scenarios'][q_idx]['problem']
                    expected_approach = assessment_data['scenarios'][q_idx].get('solution_approach', "A well-reasoned approach demonstrating understanding.")
                elif key == 'design_challenge':
                    question_text = assessment_data['design_challenge']['challenge']
                    expected_approach = "A creative and feasible design meeting criteria."
                
                if platform.llm:
                    with st.spinner(f"Generating feedback for {key.replace('_', ' ').title()}..."):
                        feedback = platform.evaluate_response(question_text, data, expected_approach)
                    st.markdown(f"#### Feedback for {key.replace('_', ' ').title()}:")
                    st.markdown(feedback)
        st.info("Assessment submitted. Scroll up to see detailed feedback.")
        # Clear current assessment data to allow generating a new one
        st.session_state.current_assessment_data = None 
        st.session_state.current_assessment_topic = None
        st.rerun()

def render_enhanced_chat_interface(topic_context, platform):
    st.subheader(f"🤖 AI Learning Assistant for {topic_context}")

    # Display existing chat messages from LangChain memory
    # The platform.memory object (ConversationBufferWindowMemory) stores messages.
    # We need to access them for display.
    if platform.memory and hasattr(platform.memory, 'chat_memory') and platform.memory.chat_memory.messages:
        for msg in platform.memory.chat_memory.messages:
            with st.chat_message(msg.type): # 'human' or 'ai'
                # Parse for image suggestions if it's an AI message
                if msg.type == "ai":
                    cleaned_response, image_queries = extract_image_suggestions(msg.content, "IMAGE_SEARCH")
                    st.markdown(cleaned_response)
                    
                    # Optional: Display images that might have been part of this historical message
                    # This part is tricky as we don't store the image URLs with the message.
                    # For simplicity, we'll only actively fetch for new messages.
                    # However, if the tag is still in cleaned_response, we could try.

                else: # Human message
                    st.markdown(msg.content)
    else:
        st.caption("No messages yet. Ask something!")

    # Chat input
    user_prompt = st.chat_input(f"Ask about {topic_context} or related concepts...")

    if user_prompt:
        # Add user message to LangChain memory and display
        # platform.memory.chat_memory.add_user_message(user_prompt) # This is done by ConversationChain
        with st.chat_message("human"):
            st.markdown(user_prompt)

        # Get AI response
        with st.chat_message("ai"):
            with st.spinner("AI is thinking..."):
                if platform.llm:
                    raw_ai_response = platform.chat_with_memory(user_prompt, topic_context)
                else:
                    raw_ai_response = "Error: LLM not available for chat."

            cleaned_response, image_queries = extract_image_suggestions(raw_ai_response, "IMAGE_SEARCH")
            st.markdown(cleaned_response)
            
            # TTS Button for AI's response
            if cleaned_response and "Error:" not in cleaned_response:
                # Unique key for TTS button
                tts_button_key = f"tts_chat_{abs(hash(cleaned_response))}_{int(time.time())}"
                if st.button("🔊 Read AI's Response", key=tts_button_key):
                    with st.spinner("Generating audio for AI response..."):
                        audio_bytes = text_to_speech_audio(cleaned_response)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

            # Display suggested images from AI's response
            if image_queries:
                st.markdown("---") # Separator for images
                for iq_idx, img_query in enumerate(image_queries):
                    st.caption(f"AI suggested visual for: *{img_query}*")
                    with st.spinner(f"Fetching image for '{img_query}'..."):
                        images = search_educational_images(img_query, num_images=1)
                    if images:
                        display_images_with_context(images, img_query, num_cols=1)
                    else:
                        st.caption(f"_No image found for: {img_query}_")
        
        # LangChain's ConversationChain should automatically handle adding AI response to memory.
        # Rerun to ensure the chat history display is updated from memory if needed,
        # but st.chat_input usually handles reruns well for new messages.
        # st.rerun() # Not always necessary, can cause blips. Test behavior.


def process_chat_input(prompt, topic, platform, context_mode):
    # This function was part of the original thought process but is now integrated
    # into render_enhanced_chat_interface directly using st.chat_input.
    # Keeping its logic in mind for the implementation above.
    pass


def render_analytics(): # Placeholder, enhance as needed
    """Enhanced learning analytics"""
    st.header("📈 Learning Analytics & Insights")
    
    # Enhanced progress tracking
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Learning Progress")
        
        # Create enhanced sample data
        progress_data = {
            'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'Study_Hours': [2, 1.5, 3, 2.5, 1, 0, 2, 3, 2, 1.5, 2.5, 3, 2, 1, 2.5, 3, 2, 1.5, 2, 3, 2.5, 1, 2, 3, 2, 1.5, 2.5, 3, 2, 1],
            'Quiz_Scores': [85, 78, 92, 88, 75, 0, 82, 95, 87, 79, 91, 94, 83, 76, 89, 96, 84, 80, 86, 93, 90, 77, 85, 97, 88, 81, 92, 95, 86, 78],
            'Engagement': [8, 7, 9, 8, 6, 0, 7, 9, 8, 7, 9, 9, 8, 6, 8, 9, 8, 7, 8, 9, 9, 7, 8, 9, 8, 7, 9, 9, 8, 7]
        }
        
        df = pd.DataFrame(progress_data)
        
        # Study hours trend
        fig_hours = px.line(df, x='Date', y='Study_Hours', title='Daily Study Hours Trend')
        fig_hours.add_scatter(x=df['Date'], y=df['Study_Hours'].rolling(7).mean(), 
                             mode='lines', name='7-day Average', line=dict(dash='dash'))
        st.plotly_chart(fig_hours, use_container_width=True)
        
        # Performance correlation
        fig_corr = px.scatter(df, x='Study_Hours', y='Quiz_Scores', 
                             size='Engagement', title='Study Time vs Performance',
                             trendline="ols")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Skills Analysis")
        
        # Enhanced skills radar
        skills_data = {
            'Skills': ['Process Design', 'Thermodynamics', 'Fluid Mechanics', 'Heat Transfer', 'Control Systems', 'Problem Solving'],
            'Current': [85, 78, 92, 88, 75, 82],
            'Target': [95, 85, 95, 92, 85, 90]
        }
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=skills_data['Current'],
            theta=skills_data['Skills'],
            fill='toself',
            name='Current Level',
            line_color='blue'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=skills_data['Target'],
            theta=skills_data['Skills'],
            fill='toself',
            name='Target Level',
            line_color='red',
            opacity=0.6
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            title="Skills Proficiency vs Targets"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Learning insights with AI
        st.subheader("🧠 AI Learning Insights")
        
        if st.button("Generate Learning Insights"):
            platform = EnhancedLearningPlatform()
            if platform.llm:
                with st.spinner("Analyzing your learning patterns..."):
                    insights_prompt = f"""
                    Analyze this learning data and provide insights:
                    
                    User Profile: {st.session_state.user_profile}
                    Learning Context: {st.session_state.learning_context}
                    Assessment Results: {len(st.session_state.assessment_results)} completed
                    
                    Provide:
                    1. Strengths and areas for improvement
                    2. Personalized learning recommendations
                    3. Next steps for skill development
                    4. Motivational insights
                    """
                    
                    insights = platform.chat_with_memory(insights_prompt)
                    st.write(insights)
    
    # Learning pathway visualization
    st.subheader("🛤️ Learning Pathway")
    
    # Create learning pathway data
    pathway_data = {
        'Stage': ['Foundation', 'Intermediate', 'Advanced', 'Expert'],
        'Process Engineering': [80, 60, 30, 10],
        'Mechanical Engineering': [90, 70, 40, 15],
        'Robotics': [70, 50, 25, 5]
    }
    
    pathway_df = pd.DataFrame(pathway_data)
    
    fig_pathway = px.bar(pathway_df, x='Stage', y=['Process Engineering', 'Mechanical Engineering', 'Robotics'],
                        title='Learning Pathway Progress', barmode='group')
    st.plotly_chart(fig_pathway, use_container_width=True)



def render_settings(): # Placeholder, enhance as needed
    st.header("⚙️ Platform Settings")
    st.info("Configure API keys, learning preferences, and data management. (Enhanced version to be built out)")
    # ... (Existing settings code can be placed here, or enhanced) ...
    with st.expander("API Keys Configuration"):
        st.write(f"Gemini API Key Status: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not Set'}")
        st.write(f"Unsplash API Key Status: {'Set' if os.getenv('UNSPLASH_ACCESS_KEY') else 'Not Set'}")
    # (User's existing settings code would go here)


def main():
    init_session_state()
    
    st.sidebar.title("🎓 Ribara Learning Platform")
    st.sidebar.markdown("*Shape Knowledge, Sharpen Skill*")
    
    # Initialize LLM and memory early to check API status
    llm, memory = configure_langchain_llm() 
    api_status = "🟢 Connected" if llm and memory else "🔴 Disconnected (Check GEMINI_API_KEY)"
    st.sidebar.write(f"**API Status:** {api_status}")
    if not llm or not memory:
        st.sidebar.warning("Core AI functionalities may be limited due to API connection issues.")
    
    # Ensure langchain_memory in session_state is the one from configure_langchain_llm
    # This is crucial for the chat memory to persist across reruns and be used by the platform
    if 'langchain_memory' not in st.session_state or st.session_state.langchain_memory is None:
        st.session_state.langchain_memory = memory

    page = st.sidebar.selectbox("Navigate to:", ["Dashboard", "Profile", "Analytics", "Settings"])
    
    if st.session_state.learning_context.get('previous_topics'):
        st.sidebar.markdown("### 🧠 Learning Context")
        st.sidebar.write(f"**Topics Explored:**")
        topics_list = st.session_state.learning_context['previous_topics'].split(', ')
        for t in topics_list[-5:]: st.sidebar.write(f"• {t.strip()}")

    if page == "Profile":
        render_user_profile()
    elif page == "Analytics":
        render_analytics() # User's existing analytics code should be here
    elif page == "Settings":
        render_settings() # User's existing settings code should be here
    else:  # Dashboard
        if st.session_state.current_module:
            render_learning_module(st.session_state.current_module)
        else:
            render_learning_dashboard()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Features:**")
    st.sidebar.markdown("🔊 Text-to-Speech")
    st.sidebar.markdown("🖼️ AI-Suggested Images")
    # ... other features

if __name__ == "__main__":
    main()