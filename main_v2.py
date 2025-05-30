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

load_dotenv()

# Configure page
st.set_page_config(
    page_title="AI Learning Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Gemini API and LangChain
@st.cache_resource
def configure_langchain_llm():
    """Configure LangChain with Google Gemini"""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Please set GEMINI_API_KEY in environment variables")
        return None, None
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Create LangChain LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20",
        google_api_key=api_key,
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Create memory for conversation
    memory = ConversationBufferWindowMemory(
        k=10,  # Keep last 10 exchanges
        return_messages=True,
        memory_key="chat_history"
    )
    
    return llm, memory

@st.cache_data
def search_educational_images(query: str, num_images: int = 3) -> List[str]:
    """Search for educational images using Unsplash API"""
    try:
        # Using Unsplash for educational images (free tier)
        # You can also use other APIs like Pixabay, Pexels, etc.
        access_key = os.getenv("UNSPLASH_ACCESS_KEY", "")
        
        if access_key:
            url = f"https://api.unsplash.com/search/photos"
            params = {
                'query': f"{query} engineering education diagram",
                'per_page': num_images,
                'orientation': 'landscape'
            }
            headers = {'Authorization': f'Client-ID {access_key}'}
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return [photo['urls']['regular'] for photo in data['results']]
        
        # Fallback to placeholder images if no API key
        placeholder_images = [
            f"https://picsum.photos/800/400?random={i}&blur=1" for i in range(num_images)
        ]
        return placeholder_images
        
    except Exception as e:
        st.warning(f"Could not fetch images: {str(e)}")
        return []

def display_images_with_context(images: List[str], context: str):
    """Display images with educational context"""
    if not images:
        return
    
    st.subheader("📸 Visual Learning Aids")
    
    cols = st.columns(min(len(images), 3))
    for idx, img_url in enumerate(images[:3]):
        with cols[idx]:
            try:
                response = requests.get(img_url, timeout=10)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption=f"Illustration {idx+1}", use_container_width=True)
            except Exception as e:
                st.error(f"Could not load image {idx+1}")

# Initialize session state
def init_session_state():
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'skill_level': 'Beginner',
            'learning_goals': [],
            'completed_assessments': [],
            'progress_data': {},
            'chat_history': []
        }
    
    if 'current_module' not in st.session_state:
        st.session_state.current_module = None
    
    if 'assessment_results' not in st.session_state:
        st.session_state.assessment_results = {}
    
    if 'langchain_memory' not in st.session_state:
        _, memory = configure_langchain_llm()
        st.session_state.langchain_memory = memory
    
    if 'learning_context' not in st.session_state:
        st.session_state.learning_context = {}

# Learning modules data structure
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
        self.setup_chains()
    
    def setup_chains(self):
        """Setup LangChain chains for different learning tasks"""
        if not self.llm:
            return
        
        # Content Generation Chain
        content_template = PromptTemplate(
            input_variables=["topic", "user_level", "learning_style", "previous_knowledge"],
            template="""
            You are an expert educational content creator. Create personalized learning content for:
            
            Topic: {topic}
            User Level: {user_level}
            Learning Style: {learning_style}
            Previous Knowledge: {previous_knowledge}
            
            Please provide:
            1. A clear explanation suitable for {user_level} level
            2. 3 practical real-world examples or applications
            3. 2 hands-on exercises with step-by-step guidance
            4. Key takeaways and summary
            5. Connection to previous topics learned
            
            Format the response with clear sections and use analogies where appropriate.
            Make it engaging and interactive.
            """
        )
        
        # Assessment Chain
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
                    {{
                        "question": "Question text",
                        "options": ["A", "B", "C", "D"],
                        "correct": 0,
                        "explanation": "Detailed explanation",
                        "difficulty": "easy/medium/hard"
                    }}
                ],
                "scenarios": [
                    {{
                        "problem": "Real-world scenario description", 
                        "context": "Background information",
                        "solution_approach": "Expected approach",
                        "key_concepts": ["concept1", "concept2"]
                    }}
                ],
                "design_challenge": {{
                    "challenge": "Design task description",
                    "constraints": ["constraint1", "constraint2"],
                    "criteria": ["criterion1", "criterion2"],
                    "deliverables": ["deliverable1", "deliverable2"]
                }}
            }}
            """
        )
        
        # Feedback Chain
        feedback_template = PromptTemplate(
            input_variables=["question", "user_answer", "correct_answer", "user_context"],
            template="""
            Evaluate this learning response with personalized feedback:
            
            Question: {question}
            User Answer: {user_answer}
            Expected Answer: {correct_answer}
            User Context: {user_context}
            
            Provide:
            1. Score (0-100) with justification
            2. Detailed constructive feedback
            3. Specific areas for improvement
            4. Personalized recommendations for next steps
            5. Encouragement and motivation
            6. Additional resources or practice suggestions
            
            Be encouraging while being specific about improvements needed.
            """
        )
        
        # Create chains
        self.content_chain = LLMChain(llm=self.llm, prompt=content_template, output_key="content")
        self.assessment_chain = LLMChain(llm=self.llm, prompt=assessment_template, output_key="assessment")
        self.feedback_chain = LLMChain(llm=self.llm, prompt=feedback_template, output_key="feedback")
        
        # Sequential chain for comprehensive learning
        self.learning_sequence = SequentialChain(
        chains=[self.content_chain, self.assessment_chain, self.feedback_chain],
        input_variables=["topic", "user_level", "learning_style", "previous_knowledge", "level", "learning_context", "question", "user_answer", "correct_answer", "user_context"],
        output_variables=["content", "assessment", "feedback"],
        verbose=True
    )
        
        # Conversational chain with memory
        conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI learning assistant specializing in engineering education. 
            You have access to the conversation history and can reference previous topics discussed.
            
            Always:
            - Provide clear, accurate technical explanations
            - Use analogies and real-world examples
            - Encourage hands-on learning
            - Reference previous learning when relevant
            - Suggest visual aids and diagrams when helpful
            - Be patient and supportive
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=conversation_prompt,
            verbose=True
        )
    
    def generate_personalized_content(self, topic, user_level, learning_style, context=""):
        """Generate personalized content using LangChain"""
        if not self.llm:
            return "Error: LangChain LLM not configured"
        
        try:
            # Get previous knowledge from context
            previous_knowledge = st.session_state.learning_context.get('previous_topics', '')
            
            response = self.content_chain.run(
                topic=topic,
                user_level=user_level,
                learning_style=learning_style,
                previous_knowledge=previous_knowledge
            )
            
            # Update learning context
            if topic not in st.session_state.learning_context.get('previous_topics', ''):
                if 'previous_topics' not in st.session_state.learning_context:
                    st.session_state.learning_context['previous_topics'] = topic
                else:
                    st.session_state.learning_context['previous_topics'] += f", {topic}"
            
            return response
        except Exception as e:
            return f"Error generating content: {str(e)}"
    
    def create_assessment(self, topic, level):
        """Create dynamic assessment using LangChain"""
        if not self.llm:
            return {"error": "LangChain LLM not configured"}
        
        try:
            learning_context = st.session_state.learning_context.get('previous_topics', '')
            
            response = self.assessment_chain.run(
                topic=topic,
                level=level,
                learning_context=learning_context
            )
            
            # Clean and parse JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Could not parse assessment JSON"}
                
        except Exception as e:
            return {"error": f"Error creating assessment: {str(e)}"}
    
    def evaluate_response(self, question, user_answer, correct_answer):
        """Evaluate response with personalized feedback using LangChain"""
        if not self.llm:
            return "Error: LangChain LLM not configured"
        
        try:
            user_context = {
                'skill_level': st.session_state.user_profile['skill_level'],
                'learning_goals': st.session_state.user_profile.get('learning_goals', []),
                'previous_topics': st.session_state.learning_context.get('previous_topics', '')
            }
            
            response = self.feedback_chain.run(
                question=question,
                user_answer=user_answer,
                correct_answer=correct_answer,
                user_context=str(user_context)
            )
            
            return response
        except Exception as e:
            return f"Error evaluating response: {str(e)}"
    
    def chat_with_memory(self, user_input, topic_context=""):
        """Enhanced chat with conversation memory and context"""
        if not self.llm:
            return "Error: LangChain LLM not configured"
        
        try:
            # Add topic context to input if provided
            enhanced_input = user_input
            if topic_context:
                enhanced_input = f"Context: We are learning about {topic_context}. Question: {user_input}"
            
            response = self.conversation_chain.predict(input=enhanced_input)
            return response
        except Exception as e:
            return f"Error in chat response: {str(e)}"

def render_user_profile():
    """Render user profile setup"""
    st.header("👤 User Profile")
    
    with st.form("profile_form"):
        name = st.text_input("Name", value=st.session_state.user_profile['name'])
        skill_level = st.selectbox(
            "Overall Skill Level",
            ['Beginner', 'Intermediate', 'Advanced'],
            index=['Beginner', 'Intermediate', 'Advanced'].index(
                st.session_state.user_profile['skill_level']
            )
        )
        
        learning_goals = st.multiselect(
            "Learning Goals",
            ['Career Advancement', 'Academic Study', 'Personal Interest', 'Certification'],
            default=st.session_state.user_profile['learning_goals']
        )
        
        learning_style = st.selectbox(
            "Preferred Learning Style",
            ['Visual', 'Auditory', 'Kinesthetic', 'Reading/Writing']
        )
        
        submitted = st.form_submit_button("Update Profile")
        
        if submitted:
            st.session_state.user_profile.update({
                'name': name,
                'skill_level': skill_level,
                'learning_goals': learning_goals,
                'learning_style': learning_style
            })
            st.success("Profile updated successfully!")

def render_learning_dashboard():
    """Render main learning dashboard"""
    st.header("📚 Learning Dashboard")
    
    # Welcome message with context
    if st.session_state.user_profile['name']:
        st.write(f"Welcome back, **{st.session_state.user_profile['name']}**! 👋")
    
    # Progress overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Modules Completed", len(st.session_state.user_profile['completed_assessments']))
    
    with col2:
        total_topics = sum(len(module['topics']) for module in LEARNING_MODULES.values())
        completed_topics = len(st.session_state.assessment_results)
        progress = (completed_topics / total_topics) * 100 if total_topics > 0 else 0
        st.metric("Overall Progress", f"{progress:.1f}%")
    
    with col3:
        st.metric("Skill Level", st.session_state.user_profile['skill_level'])
    
    with col4:
        context_topics = st.session_state.learning_context.get('previous_topics', 'None')
        st.metric("Topics Explored", len(context_topics.split(', ')) if context_topics != 'None' else 0)
    
    # Learning path recommendation
    if st.session_state.learning_context.get('previous_topics'):
        st.info(f"🎯 **Learning Context**: You've been exploring {st.session_state.learning_context['previous_topics']}")
    
    # Learning modules
    st.subheader("Available Learning Modules")
    
    cols = st.columns(len(LEARNING_MODULES))
    for idx, (key, module) in enumerate(LEARNING_MODULES.items()):
        with cols[idx]:
            with st.container():
                st.write(f"**{module['title']}**")
                st.write(module['description'])
                st.write(f"Topics: {len(module['topics'])}")
                
                if st.button(f"Start Learning", key=f"start_{key}"):
                    st.session_state.current_module = key
                    st.rerun()

def render_learning_module(module_key):
    """Render individual learning module with enhanced features"""
    module = LEARNING_MODULES[module_key]
    platform = EnhancedLearningPlatform()
    
    st.header(f"📖 {module['title']}")
    
    # Module navigation
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("← Back to Dashboard"):
            st.session_state.current_module = None
            st.rerun()
    
    # Topic selection
    selected_topic = st.selectbox("Select Topic", module['topics'])
    
    # Learning content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Learn", "🖼️ Visual", "🧪 Practice", "📊 Assessment", "💬 AI Tutor"])
    
    with tab1:
        st.subheader("Personalized Learning Content")
        
        learning_style = st.radio(
            "Learning Approach",
            ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"],
            horizontal=True
        )
        
        if st.button("Generate Personalized Content", type="primary"):
            with st.spinner("Creating personalized learning experience..."):
                content = platform.generate_personalized_content(
                    selected_topic,
                    st.session_state.user_profile['skill_level'],
                    learning_style
                )
                
                st.markdown("### 📚 Learning Material")
                st.write(content)
                
                # Store the content for context
                st.session_state.learning_context[f"{selected_topic}_content"] = content
    
    with tab2:
        st.subheader("🖼️ Visual Learning Aids")
        
        col1, col2 = st.columns([2, 1])
        with col2:
            if st.button("Find Educational Images", type="secondary"):
                with st.spinner("Searching for relevant images..."):
                    images = search_educational_images(f"{selected_topic} {module['title']}")
                    if images:
                        st.session_state[f"images_{selected_topic}"] = images
        
        # Display images if available
        if f"images_{selected_topic}" in st.session_state:
            images = st.session_state[f"images_{selected_topic}"]
            display_images_with_context(images, selected_topic)
            
            # Generate image explanations
            if st.button("Explain These Images"):
                with st.spinner("Generating image explanations..."):
                    explanation_prompt = f"Explain how these images relate to {selected_topic} in {module['title']}. Provide educational context."
                    explanation = platform.chat_with_memory(explanation_prompt, selected_topic)
                    st.markdown("### 🔍 Image Context & Explanation")
                    st.write(explanation)
    
    with tab3:
        st.subheader("🧪 Interactive Practice")
        
        # Enhanced exercises with context
        st.markdown("### Hands-on Learning Exercise")
        
        exercise_type = st.selectbox(
            "Choose Exercise Type",
            ["Problem Solving", "Design Challenge", "Case Study", "Calculation Practice"]
        )
        
        if st.button("Generate Practice Exercise"):
            with st.spinner("Creating personalized exercise..."):
                exercise_prompt = f"""
                Create a {exercise_type.lower()} exercise for {selected_topic} 
                at {st.session_state.user_profile['skill_level']} level.
                
                Include:
                1. Clear problem statement
                2. Required background knowledge
                3. Step-by-step approach hints
                4. Expected deliverables
                """
                
                exercise = platform.chat_with_memory(exercise_prompt, selected_topic)
                st.markdown("### 📋 Your Exercise")
                st.write(exercise)
        
        # Exercise submission
        st.markdown("### Submit Your Solution")
        exercise_answer = st.text_area(
            "Your solution/approach:",
            height=150,
            placeholder="Explain your approach, show calculations, or describe your design..."
        )
        
        if st.button("Submit & Get Feedback") and exercise_answer:
            with st.spinner("Analyzing your solution..."):
                feedback = platform.evaluate_response(
                    f"{exercise_type} for {selected_topic}",
                    exercise_answer,
                    "Comprehensive solution with proper methodology"
                )
                st.markdown("### 📝 Personalized Feedback")
                st.write(feedback)
    
    with tab4:
        st.subheader("📊 Adaptive Assessment")
        
        # Assessment difficulty selection
        difficulty = st.selectbox(
            "Assessment Difficulty",
            ["Basic Understanding", "Applied Knowledge", "Advanced Problem Solving"]
        )
        
        if st.button("Generate Smart Assessment"):
            with st.spinner("Creating adaptive assessment..."):
                assessment = platform.create_assessment(selected_topic, difficulty)
                
                if 'error' not in assessment:
                    st.session_state.current_assessment = assessment
                    st.session_state.assessment_topic = selected_topic
                    st.rerun()
                else:
                    st.error(assessment['error'])
        
        # Display assessment if available
        if hasattr(st.session_state, 'current_assessment'):
            render_enhanced_assessment(st.session_state.current_assessment, selected_topic, platform)
    
    with tab5:
        render_enhanced_chat_interface(selected_topic, platform)

def render_enhanced_assessment(assessment, topic, platform):
    """Render enhanced interactive assessment with better feedback"""
    st.write("### 🎯 Adaptive Assessment")
    
    user_answers = {}
    
    # Multiple choice questions with enhanced display
    if 'multiple_choice' in assessment:
        st.markdown("#### Multiple Choice Questions")
        for i, q in enumerate(assessment['multiple_choice']):
            with st.expander(f"Question {i+1} - {q.get('difficulty', 'medium').title()}", expanded=True):
                st.write(f"**{q['question']}**")
                answer = st.radio(
                    "Select your answer:",
                    q['options'],
                    key=f"mc_{i}"
                )
                user_answers[f"mc_{i}"] = {
                    'answer': answer,
                    'question': q['question'],
                    'correct': q['options'][q['correct']],
                    'explanation': q.get('explanation', '')
                }
    
    # Scenario questions with better context
    if 'scenarios' in assessment:
        st.markdown("#### Real-World Scenarios")
        for i, scenario in enumerate(assessment['scenarios']):
            with st.expander(f"Scenario {i+1}", expanded=True):
                st.write(f"**Context:** {scenario.get('context', '')}")
                st.write(f"**Problem:** {scenario['problem']}")
                if 'key_concepts' in scenario:
                    st.write(f"**Key Concepts:** {', '.join(scenario['key_concepts'])}")
                
                scenario_answer = st.text_area(
                    "Your detailed approach:",
                    key=f"scenario_{i}",
                    height=120
                )
                user_answers[f"scenario_{i}"] = {
                    'answer': scenario_answer,
                    'problem': scenario['problem'],
                    'expected': scenario.get('solution_approach', '')
                }
    
    # Design challenge with detailed criteria
    if 'design_challenge' in assessment:
        st.markdown("#### Design Challenge")
        challenge = assessment['design_challenge']
        
        with st.expander("Design Challenge", expanded=True):
            st.write(f"**Challenge:** {challenge['challenge']}")
            if 'constraints' in challenge:
                st.write(f"**Constraints:** {', '.join(challenge['constraints'])}")
            if 'criteria' in challenge:
                st.write(f"**Evaluation Criteria:** {', '.join(challenge['criteria'])}")
            
            design_answer = st.text_area(
                "Your design solution:",
                key="design_challenge",
                height=150
            )
            user_answers['design_challenge'] = {
                'answer': design_answer,
                'challenge': challenge['challenge']
            }
    
    # Enhanced submission and feedback
    if st.button("Submit Assessment", type="primary"):
        with st.spinner("Analyzing your responses..."):
            # Store results
            st.session_state.assessment_results[topic] = user_answers
            
            # Calculate detailed scores
            if 'multiple_choice' in assessment:
                correct = 0
                total = len(assessment['multiple_choice'])
                
                st.markdown("### 📊 Detailed Results")
                
                for i, q in enumerate(assessment['multiple_choice']):
                    user_ans = user_answers.get(f"mc_{i}", {})
                    is_correct = user_ans.get('answer') == q['options'][q['correct']]
                    if is_correct:
                        correct += 1
                    
                    # Show individual question feedback
                    status = "✅" if is_correct else "❌"
                    with st.expander(f"{status} Question {i+1} - {q.get('difficulty', 'medium').title()}"):
                        st.write(f"**Your answer:** {user_ans.get('answer', 'No answer')}")
                        st.write(f"**Correct answer:** {q['options'][q['correct']]}")
                        st.write(f"**Explanation:** {q.get('explanation', 'No explanation provided')}")
                
                score = (correct / total) * 100
                st.metric("Multiple Choice Score", f"{score:.1f}%")
                
                # Personalized feedback for scenarios
                for key, data in user_answers.items():
                    if key.startswith('scenario_') or key == 'design_challenge':
                        if data.get('answer'):
                            feedback = platform.evaluate_response(
                                data.get('problem', data.get('challenge', '')),
                                data['answer'],
                                data.get('expected', 'Comprehensive solution')
                            )
                            st.markdown(f"### 💬 Feedback for {key.replace('_', ' ').title()}")
                            st.write(feedback)
        
        st.success("Assessment completed! Check your detailed feedback above.")

def render_enhanced_chat_interface(topic, platform):
    """Enhanced AI chat with memory and context"""
    st.subheader("🤖 AI Learning Assistant")
    
    # Chat context controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear Memory", help="Start fresh conversation"):
            platform.memory.clear()
            st.success("Conversation memory cleared!")
    
    with col2:
        if st.button("Show Context", help="View conversation context"):
            if hasattr(platform.memory, 'chat_memory'):
                st.write("**Conversation History:**")
                for message in platform.memory.chat_memory.messages[-6:]:  # Show last 6 messages
                    role = "You" if isinstance(message, HumanMessage) else "AI"
                    st.write(f"**{role}:** {message.content[:100]}...")
    
    with col3:
        context_mode = st.checkbox("Enhanced Context Mode", value=True, 
                                   help="Use topic context for better responses")
    
    # Display chat messages from memory
    st.markdown("### 💬 Conversation")
    
    if hasattr(platform.memory, 'chat_memory') and platform.memory.chat_memory.messages:
        for message in platform.memory.chat_memory.messages[-10:]:  # Show last 10 messages
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.write(message.content)
    
    # Enhanced chat input with suggestions
    st.markdown("### 🎯 Quick Questions")
    quick_questions = [
        f"Explain {topic} with a real-world example",
        f"What are common mistakes in {topic}?",
        f"How does {topic} connect to other engineering concepts?",
        f"Give me a practice problem for {topic}",
        f"What tools or software are used for {topic}?"
    ]
    
    selected_question = st.selectbox("Or choose a quick question:", ["Custom question..."] + quick_questions)
    
    # Chat input
    if selected_question != "Custom question...":
        prompt = selected_question
        if st.button("Ask Selected Question"):
            process_chat_input(prompt, topic, platform, context_mode)
    else:
        prompt = st.chat_input(f"Ask me anything about {topic}...")
        if prompt:
            process_chat_input(prompt, topic, platform, context_mode)

def process_chat_input(prompt, topic, platform, context_mode):
    """Process chat input with enhanced context"""
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if context_mode:
                response = platform.chat_with_memory(prompt, topic)
            else:
                response = platform.chat_with_memory(prompt)
            
            st.write(response)
            
            # Suggest follow-up questions based on response
            if "error" not in response.lower():
                follow_up_prompt = f"Based on our discussion about {topic}, suggest 2 relevant follow-up questions a student might ask."
                try:
                    follow_ups = platform.chat_with_memory(follow_up_prompt, topic)
                    with st.expander("💡 Suggested Follow-up Questions"):
                        st.write(follow_ups)
                except:
                    pass  # Silently fail if follow-up generation fails

def render_analytics():
    """Enhanced learning analytics with LangChain insights"""
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

def render_settings():
    """Enhanced settings with LangChain configurations"""
    st.header("⚙️ Platform Settings")
    
    # API Configuration
    st.subheader("🔧 API Configuration")
    
    with st.expander("API Keys Configuration"):
        # Gemini API Key
        current_gemini = "Set" if os.getenv("GEMINI_API_KEY") else "Not Set"
        st.write(f"**Gemini API Key:** {current_gemini}")
        
        # Unsplash API Key (for images)
        current_unsplash = "Set" if os.getenv("UNSPLASH_ACCESS_KEY") else "Not Set"
        st.write(f"**Unsplash API Key:** {current_unsplash}")
        
        st.info("💡 Add API keys to Streamlit secrets for full functionality")
        
        if st.button("Test API Connections"):
            platform = EnhancedLearningPlatform()
            if platform.llm:
                st.success("✅ Gemini API: Connected")
            else:
                st.error("❌ Gemini API: Not Connected")
            
            try:
                test_images = search_educational_images("engineering", 1)
                if test_images:
                    st.success("✅ Image Search: Connected")
                else:
                    st.warning("⚠️ Image Search: Limited (using placeholders)")
            except:
                st.warning("⚠️ Image Search: Limited (using placeholders)")
    
    # Learning Preferences
    st.subheader("🎯 Learning Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Memory settings
        memory_length = st.slider("Conversation Memory Length", 5, 20, 10,
                                  help="Number of previous exchanges to remember")
        
        # Content generation settings
        content_complexity = st.selectbox(
            "Default Content Complexity",
            ["Simplified", "Standard", "Detailed", "Comprehensive"]
        )
        
        # Assessment settings
        assessment_difficulty = st.selectbox(
            "Default Assessment Difficulty",
            ["Basic", "Standard", "Challenging", "Expert"]
        )
    
    with col2:
        # Visual preferences
        show_images = st.checkbox("Auto-load Educational Images", True)
        
        # Feedback preferences
        detailed_feedback = st.checkbox("Detailed Assessment Feedback", True)
        
        # Chat preferences
        show_follow_ups = st.checkbox("Show Follow-up Suggestions", True)
        
        # Analytics preferences
        auto_insights = st.checkbox("Auto-generate Learning Insights", False)
    
    # Export/Import Settings
    st.subheader("💾 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Learning Data"):
            export_data = {
                'user_profile': st.session_state.user_profile,
                'learning_context': st.session_state.learning_context,
                'assessment_results': st.session_state.assessment_results
            }
            
            json_data = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="learning_data.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Reset All Data"):
            if st.checkbox("I understand this will delete all my progress"):
                for key in ['user_profile', 'learning_context', 'assessment_results', 'langchain_memory']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("All data reset successfully!")
                st.rerun()
    
    with col3:
        st.info("🔄 Data is automatically saved during your session")
    
    # Advanced Settings
    with st.expander("🔬 Advanced Settings"):
        st.write("**LangChain Configuration:**")
        
        # Temperature setting
        llm_temperature = st.slider("AI Response Creativity", 0.0, 1.0, 0.7,
                                   help="Higher values make responses more creative")
        
        # Chain verbosity
        chain_verbose = st.checkbox("Enable Chain Debugging", False,
                                   help="Show detailed chain execution logs")
        
        # Memory type selection
        memory_type = st.selectbox(
            "Memory Type",
            ["Buffer Window", "Summary Buffer", "Conversation Summary"],
            help="How conversation history is managed"
        )
        
        if st.button("Apply Advanced Settings"):
            st.info("Advanced settings would be applied here (requires restart)")

def main():
    """Enhanced main application function"""
    init_session_state()
    
    # Sidebar navigation
    st.sidebar.title("🎓 Enhanced AI Learning Platform")
    st.sidebar.markdown("*Powered by LangChain & Gemini*")
    
    # API Status indicator
    _, memory = configure_langchain_llm()
    api_status = "🟢 Connected" if memory else "🔴 Disconnected"
    st.sidebar.write(f"**API Status:** {api_status}")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Dashboard", "Profile", "Analytics", "Settings"]
    )
    
    # Learning context display
    if st.session_state.learning_context.get('previous_topics'):
        st.sidebar.markdown("### 🧠 Learning Context")
        st.sidebar.write(f"**Topics Explored:**")
        topics = st.session_state.learning_context['previous_topics'].split(', ')
        for topic in topics[-3:]:  # Show last 3 topics
            st.sidebar.write(f"• {topic}")
    
    # Main content area
    if page == "Profile":
        render_user_profile()
    elif page == "Analytics":
        render_analytics()
    elif page == "Settings":
        render_settings()
    else:  # Dashboard
        if st.session_state.current_module:
            render_learning_module(st.session_state.current_module)
        else:
            render_learning_dashboard()
    
    # Enhanced footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚀 Enhanced Features:**")
    st.sidebar.markdown("✅ LangChain Integration")
    st.sidebar.markdown("✅ Conversation Memory")
    st.sidebar.markdown("✅ Visual Learning Aids")
    st.sidebar.markdown("✅ Adaptive Assessments")
    st.sidebar.markdown("✅ AI-Powered Insights")
    st.sidebar.markdown("✅ Personalized Feedback")
    
    # Quick stats
    if st.session_state.user_profile['name']:
        st.sidebar.markdown("### 📊 Quick Stats")
        st.sidebar.metric("Learning Sessions", len(st.session_state.learning_context))
        st.sidebar.metric("Topics Mastered", len(st.session_state.assessment_results))

if __name__ == "__main__":
    main()