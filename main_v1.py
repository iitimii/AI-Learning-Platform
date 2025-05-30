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
from dotenv import load_dotenv
import os


load_dotenv()

st.set_page_config(
    page_title="Ribara AI Learning Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Please set GEMINI_API_KEY")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

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

class LearningPlatform:
    def __init__(self):
        self.model = configure_gemini()
    
    def generate_personalized_content(self, topic, user_level, learning_style):
        """Generate personalized learning content using Gemini"""
        if not self.model:
            return "Error: Gemini model not configured"
        
        prompt = f"""
        Create personalized learning content for:
        Topic: {topic}
        User Level: {user_level}
        Learning Style: {learning_style}
        
        Please provide:
        1. A clear explanation suitable for {user_level} level
        2. 3 practical examples or applications
        3. 2 hands-on exercises
        4. Key takeaways
        
        Format the response in a structured way with clear sections.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating content: {str(e)}"
    
    def create_assessment(self, topic, level):
        """Create dynamic assessment based on topic and level"""
        prompt = f"""
        Create a comprehensive assessment for {topic} at {level} level.
        
        Generate:
        1. 5 multiple choice questions with 4 options each
        2. 2 practical problem-solving scenarios
        3. 1 design challenge
        
        Format as JSON with the following structure:
        {{
            "multiple_choice": [
                {{
                    "question": "Question text",
                    "options": ["A", "B", "C", "D"],
                    "correct": 0,
                    "explanation": "Why this is correct"
                }}
            ],
            "scenarios": [
                {{
                    "problem": "Scenario description",
                    "solution_approach": "How to approach this"
                }}
            ],
            "design_challenge": {{
                "challenge": "Design task description",
                "criteria": ["Criterion 1", "Criterion 2"]
            }}
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Parse JSON response
            return json.loads(response.text.replace('```json', '').replace('```', ''))
        except Exception as e:
            return {"error": f"Error creating assessment: {str(e)}"}
    
    def evaluate_response(self, question, user_answer, correct_answer):
        """Evaluate user response and provide feedback"""
        prompt = f"""
        Evaluate this learning response:
        Question: {question}
        User Answer: {user_answer}
        Correct Answer: {correct_answer}
        
        Provide:
        1. Score (0-100)
        2. Detailed feedback
        3. Areas for improvement
        4. Additional resources or tips
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error evaluating response: {str(e)}"

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
        
        submitted = st.form_submit_button("Update Profile")
        
        if submitted:
            st.session_state.user_profile.update({
                'name': name,
                'skill_level': skill_level,
                'learning_goals': learning_goals
            })
            st.success("Profile updated successfully!")

def render_learning_dashboard():
    """Render main learning dashboard"""
    st.header("📚 Learning Dashboard")
    
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
        st.metric("Learning Streak", "5 days")  # Placeholder
    
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
    """Render individual learning module"""
    module = LEARNING_MODULES[module_key]
    platform = LearningPlatform()
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Learn", "🧪 Practice", "📊 Assessment", "💬 Chat"])
    
    with tab1:
        st.subheader("Learning Content")
        
        learning_style = st.radio(
            "Preferred Learning Style",
            ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"]
        )
        
        if st.button("Generate Personalized Content"):
            with st.spinner("Generating personalized content..."):
                content = platform.generate_personalized_content(
                    selected_topic,
                    st.session_state.user_profile['skill_level'],
                    learning_style
                )
                st.write(content)
    
    with tab2:
        st.subheader("Interactive Exercises")
        
        # Placeholder for interactive exercises
        st.info("🚧 Interactive 3D simulations and problem-solving exercises coming soon!")
        
        # Simple exercise example
        st.write("**Quick Exercise:**")
        exercise_answer = st.text_area("Solve this problem: How would you optimize a heat exchanger for maximum efficiency?")
        
        if st.button("Submit Exercise") and exercise_answer:
            with st.spinner("Evaluating your response..."):
                feedback = platform.evaluate_response(
                    "Heat exchanger optimization",
                    exercise_answer,
                    "Consider heat transfer coefficient, pressure drop, and cost factors"
                )
                st.write("**Feedback:**")
                st.write(feedback)
    
    with tab3:
        st.subheader("Skill Assessment")
        
        if st.button("Generate Assessment"):
            with st.spinner("Creating personalized assessment..."):
                assessment = platform.create_assessment(selected_topic, st.session_state.user_profile['skill_level'])
                
                if 'error' not in assessment:
                    st.session_state.current_assessment = assessment
                    st.rerun()
                else:
                    st.error(assessment['error'])
        
        # Display assessment if available
        if hasattr(st.session_state, 'current_assessment'):
            render_assessment(st.session_state.current_assessment, selected_topic)
    
    with tab4:
        render_chat_interface(selected_topic)

def render_assessment(assessment, topic):
    """Render interactive assessment"""
    st.write("### Assessment Questions")
    
    user_answers = {}
    
    # Multiple choice questions
    if 'multiple_choice' in assessment:
        for i, q in enumerate(assessment['multiple_choice']):
            st.write(f"**Question {i+1}:** {q['question']}")
            answer = st.radio(
                "Select your answer:",
                q['options'],
                key=f"mc_{i}"
            )
            user_answers[f"mc_{i}"] = answer
    
    # Scenario questions
    if 'scenarios' in assessment:
        for i, scenario in enumerate(assessment['scenarios']):
            st.write(f"**Scenario {i+1}:** {scenario['problem']}")
            scenario_answer = st.text_area(
                "Your approach:",
                key=f"scenario_{i}"
            )
            user_answers[f"scenario_{i}"] = scenario_answer
    
    if st.button("Submit Assessment"):
        # Store results
        st.session_state.assessment_results[topic] = user_answers
        st.success("Assessment submitted! Results will be processed.")
        
        # Calculate basic score for MC questions
        if 'multiple_choice' in assessment:
            correct = 0
            total = len(assessment['multiple_choice'])
            for i, q in enumerate(assessment['multiple_choice']):
                if user_answers.get(f"mc_{i}") == q['options'][q['correct']]:
                    correct += 1
            
            score = (correct / total) * 100
            st.metric("Score", f"{score:.1f}%")

def render_chat_interface(topic):
    """Render AI chat interface"""
    st.subheader("💬 AI Learning Assistant")
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input(f"Ask me anything about {topic}..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Generate AI response
        platform = LearningPlatform()
        if platform.model:
            try:
                enhanced_prompt = f"""
                You are an expert learning assistant for {topic}. 
                The user's skill level is {st.session_state.user_profile['skill_level']}.
                
                User question: {prompt}
                
                Provide a helpful, educational response that:
                1. Directly answers their question
                2. Provides relevant examples
                3. Suggests next learning steps
                4. Maintains an encouraging tone
                """
                
                response = platform.model.generate_content(enhanced_prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

def render_analytics():
    """Render learning analytics and progress tracking"""
    st.header("📈 Learning Analytics")
    
    # Create sample data for demonstration
    progress_data = {
        'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
        'Study_Hours': [2, 1.5, 3, 2.5, 1, 0, 2, 3, 2, 1.5, 2.5, 3, 2, 1, 2.5, 3, 2, 1.5, 2, 3, 2.5, 1, 2, 3, 2, 1.5, 2.5, 3, 2, 1],
        'Quiz_Scores': [85, 78, 92, 88, 75, 0, 82, 95, 87, 79, 91, 94, 83, 76, 89, 96, 84, 80, 86, 93, 90, 77, 85, 97, 88, 81, 92, 95, 86, 78]
    }
    
    df = pd.DataFrame(progress_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Study hours chart
        fig_hours = px.line(df, x='Date', y='Study_Hours', title='Daily Study Hours')
        st.plotly_chart(fig_hours, use_container_width=True)
    
    with col2:
        # Quiz scores chart
        fig_scores = px.line(df, x='Date', y='Quiz_Scores', title='Quiz Performance Over Time')
        st.plotly_chart(fig_scores, use_container_width=True)
    
    # Skills radar chart
    skills_data = {
        'Skills': ['Process Design', 'Thermodynamics', 'Fluid Mechanics', 'Heat Transfer', 'Control Systems'],
        'Proficiency': [85, 78, 92, 88, 75]
    }
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=skills_data['Proficiency'],
        theta=skills_data['Skills'],
        fill='toself',
        name='Current Proficiency'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Skills Proficiency Radar"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

def main():
    """Main application function"""
    init_session_state()
    
    # Sidebar navigation
    st.sidebar.title("🎓 AI Learning Platform")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Dashboard", "Profile", "Analytics", "Settings"]
    )
    
    # Main content area
    if page == "Profile":
        render_user_profile()
    
    elif page == "Analytics":
        render_analytics()
    
    elif page == "Settings":
        st.header("⚙️ Settings")
        st.info("Settings panel - Configure notifications, preferences, etc.")
    
    else:  # Dashboard
        if st.session_state.current_module:
            render_learning_module(st.session_state.current_module)
        else:
            render_learning_dashboard()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Platform Features:**")
    st.sidebar.markdown("✅ Personalized AI content")
    st.sidebar.markdown("✅ Dynamic assessments")
    st.sidebar.markdown("✅ Progress tracking")
    st.sidebar.markdown("✅ Interactive chat")
    st.sidebar.markdown("🚧 3D simulations (coming soon)")
    st.sidebar.markdown("🚧 Video interactions (coming soon)")

if __name__ == "__main__":
    main()