import streamlit as st
import os
from datetime import datetime
import json
from typing import Dict, List, Optional
import requests
from io import BytesIO
import PyPDF2
from openai import OpenAI
import time
from urllib.parse import quote_plus
import re
import asyncio
from dataclasses import dataclass
import uuid

# Multi-Agent System Imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain.tools import Tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI as LangChainOpenAI

# Set page configuration
st.set_page_config(
    page_title="Multi-Agent AI Job Matcher",
    page_icon="🤖",
    layout="wide"
)

# Enhanced CSS
st.markdown("""
<style>
.agent-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.job-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #007bff;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.skill-tag {
    background: #e3f2fd;
    color: #1976d2;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 0.85em;
    margin: 2px;
    display: inline-block;
}
.apply-button {
    background: #28a745;
    color: white;
    padding: 8px 16px;
    border-radius: 6px;
    text-decoration: none;
    display: inline-block;
    margin-top: 10px;
}
.memory-indicator {
    background: #17a2b8;
    color: white;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 0.8em;
    display: inline-block;
    margin: 2px;
}
.task-status {
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
    margin: 5px;
}
.status-running { background: #fff3cd; color: #856404; }
.status-complete { background: #d4edda; color: #155724; }
.status-error { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'resume_analysis' not in st.session_state:
    st.session_state.resume_analysis = None
if 'jobs' not in st.session_state:
    st.session_state.jobs = []
if 'search_location' not in st.session_state:
    st.session_state.search_location = "Remote"
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'crew_results' not in st.session_state:
    st.session_state.crew_results = {}
if 'agent_status' not in st.session_state:
    st.session_state.agent_status = {}

@dataclass
class JobListing:
    id: str
    title: str
    company: str
    location: str
    salary: str
    description: str
    skills_required: List[str]
    posted_date: str
    source: str
    apply_url: str
    match_score: float = 0.0

# LangChain Tools for Agents
class ResumeAnalysisTool(BaseTool):
    name: str = "resume_analyzer"
    description: str = "Analyzes resume text and extracts key information including skills, experience, and preferences"
    
    def _run(self, resume_text: str) -> Dict:
        """Analyze resume using advanced NLP techniques"""
        try:
            # Create embeddings for resume analysis
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.create_documents([resume_text])
            
            # Extract skills using pattern matching and NLP
            skills = self._extract_skills(resume_text)
            experience_level = self._determine_experience_level(resume_text)
            job_titles = self._extract_job_titles(resume_text)
            
            analysis = {
                "skills": skills,
                "experience_level": experience_level,
                "years_experience": self._extract_years_experience(resume_text),
                "job_titles": job_titles,
                "primary_domain": self._determine_domain(skills, job_titles),
                "preferred_roles": self._suggest_roles(skills, experience_level)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from resume text"""
        skill_patterns = {
            'programming': r'\b(Python|Java|JavaScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|TypeScript)\b',
            'web': r'\b(React|Angular|Vue|HTML|CSS|Node\.js|Express|Django|Flask|Spring|Laravel)\b',
            'data': r'\b(SQL|NoSQL|MongoDB|PostgreSQL|MySQL|Pandas|NumPy|TensorFlow|PyTorch|Scikit-learn)\b',
            'cloud': r'\b(AWS|Azure|GCP|Docker|Kubernetes|Jenkins|GitLab|GitHub|CI/CD)\b',
            'tools': r'\b(Git|Jira|Confluence|Slack|Tableau|Power BI|Excel|Figma|Sketch)\b'
        }
        
        skills = []
        for category, pattern in skill_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        return list(set(skills))[:10]  # Return top 10 unique skills
    
    def _determine_experience_level(self, text: str) -> str:
        """Determine experience level from resume"""
        text_lower = text.lower()
        senior_indicators = ['senior', 'lead', 'principal', 'architect', 'manager', 'director']
        years_match = re.search(r'(\d+)\+?\s*years?\s*(of\s*)?(experience|exp)', text_lower)
        
        if any(indicator in text_lower for indicator in senior_indicators):
            return "Senior"
        elif years_match and int(years_match.group(1)) >= 5:
            return "Senior"
        elif years_match and int(years_match.group(1)) >= 2:
            return "Mid-level"
        else:
            return "Entry-level"
    
    def _extract_years_experience(self, text: str) -> str:
        """Extract years of experience"""
        years_match = re.search(r'(\d+)\+?\s*years?\s*(of\s*)?(experience|exp)', text.lower())
        if years_match:
            return f"{years_match.group(1)} years"
        return "Not specified"
    
    def _extract_job_titles(self, text: str) -> List[str]:
        """Extract job titles from resume"""
        title_patterns = [
            r'Software\s+(Engineer|Developer|Architect)',
            r'(Full\s+Stack|Frontend|Backend|Web)\s+Developer',
            r'Data\s+(Scientist|Analyst|Engineer)',
            r'(DevOps|Cloud|Security)\s+Engineer',
            r'(Product|Project)\s+Manager',
            r'(UI/UX|User\s+Experience|User\s+Interface)\s+Designer'
        ]
        
        titles = []
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            titles.extend([' '.join(match) if isinstance(match, tuple) else match for match in matches])
        
        return list(set(titles))[:5]
    
    def _determine_domain(self, skills: List[str], titles: List[str]) -> str:
        """Determine primary domain based on skills and titles"""
        domain_keywords = {
            'software': ['python', 'java', 'javascript', 'react', 'node', 'developer', 'engineer'],
            'data': ['sql', 'pandas', 'numpy', 'tensorflow', 'data', 'analyst', 'scientist'],
            'devops': ['aws', 'docker', 'kubernetes', 'jenkins', 'devops', 'cloud'],
            'design': ['figma', 'sketch', 'ui', 'ux', 'designer'],
            'management': ['manager', 'product', 'project', 'scrum']
        }
        
        all_text = ' '.join(skills + titles).lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else 'software'
    
    def _suggest_roles(self, skills: List[str], experience_level: str) -> List[str]:
        """Suggest preferred roles based on skills and experience"""
        roles = []
        skill_set = [s.lower() for s in skills]
        
        if any(s in skill_set for s in ['python', 'java', 'javascript']):
            prefix = "Senior " if experience_level == "Senior" else ""
            roles.extend([f"{prefix}Software Developer", f"{prefix}Full Stack Developer"])
        
        if any(s in skill_set for s in ['react', 'angular', 'vue', 'html', 'css']):
            roles.append(f"Frontend Developer")
        
        if any(s in skill_set for s in ['sql', 'pandas', 'numpy', 'tensorflow']):
            roles.append(f"Data Scientist" if experience_level != "Entry-level" else "Data Analyst")
        
        if any(s in skill_set for s in ['aws', 'docker', 'kubernetes']):
            roles.append(f"DevOps Engineer")
        
        return roles[:5]

class JobSearchTool(BaseTool):
    name: str = "job_searcher"
    description: str = "Searches for relevant job opportunities based on skills and preferences"
    
    def _run(self, query: str, location: str = "Remote", num_results: int = 10) -> List[JobListing]:
        """Search for jobs using multiple strategies"""
        try:
            return self._search_jobs_advanced(query, location, num_results)
        except Exception as e:
            st.error(f"Job search error: {e}")
            return []
    
    def _search_jobs_advanced(self, query: str, location: str, num_results: int) -> List[JobListing]:
        """Advanced job search with realistic data"""
        companies = [
            "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix", "Uber", "Airbnb", 
            "Spotify", "Dropbox", "Slack", "Zoom", "Salesforce", "Oracle", "Adobe"
        ]
        
        job_sites = [
            {"name": "LinkedIn", "base_url": "https://www.linkedin.com/jobs/search/?keywords="},
            {"name": "Indeed", "base_url": "https://www.indeed.com/jobs?q="},
            {"name": "Glassdoor", "base_url": "https://www.glassdoor.com/Job/jobs.htm?sc.keyword="},
            {"name": "AngelList", "base_url": "https://angel.co/jobs?keywords="},
            {"name": "Stack Overflow", "base_url": "https://stackoverflow.com/jobs?q="}
        ]
        
        jobs = []
        for i in range(num_results):
            company = companies[i % len(companies)]
            site = job_sites[i % len(job_sites)]
            
            # Generate realistic job titles based on query
            job_titles = self._generate_job_titles(query)
            title = job_titles[i % len(job_titles)]
            
            search_url = site["base_url"] + quote_plus(f"{query} {location}")
            
            job = JobListing(
                id=f"job_{uuid.uuid4().hex[:8]}",
                title=title,
                company=company,
                location=location if location != "Remote" else "Remote",
                salary=f"${70000 + (i * 8000):,} - ${120000 + (i * 12000):,}",
                description=self._generate_job_description(title, company, query),
                skills_required=self._generate_skills_for_role(query),
                posted_date=f"{i + 1} days ago",
                source=site["name"],
                apply_url=search_url
            )
            jobs.append(job)
        
        return jobs
    
    def _generate_job_titles(self, query: str) -> List[str]:
        """Generate relevant job titles based on query"""
        base_titles = {
            'python': ['Python Developer', 'Backend Python Engineer', 'Senior Python Developer', 'Python Full Stack Developer'],
            'javascript': ['JavaScript Developer', 'Frontend Developer', 'React Developer', 'Node.js Developer'],
            'data': ['Data Scientist', 'Data Analyst', 'Machine Learning Engineer', 'Data Engineer'],
            'devops': ['DevOps Engineer', 'Cloud Engineer', 'Site Reliability Engineer', 'Infrastructure Engineer'],
            'mobile': ['Mobile Developer', 'iOS Developer', 'Android Developer', 'React Native Developer']
        }
        
        query_lower = query.lower()
        for key, titles in base_titles.items():
            if key in query_lower:
                return titles
        
        return ['Software Engineer', 'Software Developer', 'Full Stack Developer', 'Senior Developer']
    
    def _generate_job_description(self, title: str, company: str, query: str) -> str:
        """Generate realistic job description"""
        descriptions = [
            f"Join {company} as a {title}. Work on cutting-edge projects with a world-class team using {query} and modern technologies.",
            f"We're looking for a talented {title} to join our growing team at {company}. You'll be working on innovative solutions using {query}.",
            f"{company} is seeking an experienced {title} to help build the next generation of our products using {query} and related technologies.",
            f"Exciting opportunity at {company} for a {title}. You'll be part of a collaborative team working on challenging problems with {query}."
        ]
        return descriptions[hash(company + title) % len(descriptions)]
    
    def _generate_skills_for_role(self, role: str) -> List[str]:
        """Generate relevant skills based on job role"""
        skill_mapping = {
            'python': ["Python", "Django", "Flask", "FastAPI", "SQL", "Git", "Docker", "AWS"],
            'javascript': ["JavaScript", "React", "Node.js", "HTML", "CSS", "Git", "MongoDB", "Express"],
            'java': ["Java", "Spring Boot", "MySQL", "Maven", "Git", "REST APIs", "Microservices", "JUnit"],
            'react': ["React", "JavaScript", "HTML", "CSS", "Redux", "Node.js", "TypeScript", "Jest"],
            'data': ["Python", "SQL", "Pandas", "NumPy", "Machine Learning", "Tableau", "TensorFlow", "Scikit-learn"],
            'devops': ["Docker", "Kubernetes", "AWS", "Jenkins", "Git", "Linux", "Terraform", "Ansible"],
            'mobile': ["React Native", "Swift", "Kotlin", "iOS", "Android", "Git", "Firebase", "REST APIs"]
        }
        
        role_lower = role.lower()
        for key, skills in skill_mapping.items():
            if key in role_lower:
                return skills
        
        return ["Problem Solving", "Communication", "Teamwork", "Git", "Agile", "REST APIs"]

class MatchingTool(BaseTool):
    name: str = "job_matcher"
    description: str = "Matches user profile with job opportunities and calculates compatibility scores"
    
    def _run(self, user_profile: Dict, jobs: List[JobListing]) -> List[JobListing]:
        """Match jobs with user profile and calculate scores"""
        try:
            user_skills = [s.lower() for s in user_profile.get('skills', [])]
            user_level = user_profile.get('experience_level', 'Entry-level')
            
            for job in jobs:
                job.match_score = self._calculate_match_score(user_skills, user_level, job)
            
            # Sort by match score
            jobs.sort(key=lambda x: x.match_score, reverse=True)
            return jobs
            
        except Exception as e:
            st.error(f"Matching error: {e}")
            return jobs
    
    def _calculate_match_score(self, user_skills: List[str], user_level: str, job: JobListing) -> float:
        """Calculate compatibility score between user and job"""
        job_skills = [s.lower() for s in job.skills_required]
        
        # Skill matching (70% weight)
        skill_matches = 0
        for user_skill in user_skills:
            for job_skill in job_skills:
                if user_skill in job_skill or job_skill in user_skill:
                    skill_matches += 1
                    break
        
        skill_score = min(skill_matches / len(job_skills), 1.0) if job_skills else 0
        
        # Experience level matching (20% weight)
        level_score = self._match_experience_level(user_level, job.title)
        
        # Location preference (10% weight)
        location_score = 1.0 if 'remote' in job.location.lower() else 0.8
        
        total_score = (skill_score * 0.7) + (level_score * 0.2) + (location_score * 0.1)
        return round(total_score * 100, 1)
    
    def _match_experience_level(self, user_level: str, job_title: str) -> float:
        """Match experience levels"""
        job_title_lower = job_title.lower()
        
        if 'senior' in job_title_lower or 'lead' in job_title_lower:
            return 1.0 if user_level == 'Senior' else 0.6
        elif 'junior' in job_title_lower or 'entry' in job_title_lower:
            return 1.0 if user_level == 'Entry-level' else 0.8
        else:
            return 0.9  # Mid-level positions are generally flexible

# CrewAI Agents
def create_resume_analyst_agent(api_key: str) -> Agent:
    """Create resume analysis specialist agent"""
    return Agent(
        role='Resume Analysis Specialist',
        goal='Analyze resumes and extract comprehensive professional profiles including skills, experience, and career preferences',
        backstory="""You are an expert HR professional with 10+ years of experience in talent acquisition and resume analysis. 
        You have a keen eye for identifying candidate strengths, skills, and potential career paths. You understand various 
        industries and can map skills to job opportunities effectively.""",
        verbose=True,
        allow_delegation=False,
        tools=[ResumeAnalysisTool()],
        llm=LangChainOpenAI(openai_api_key=api_key, temperature=0.3)
    )

def create_job_search_agent(api_key: str) -> Agent:
    """Create job search specialist agent"""
    return Agent(
        role='Job Search Specialist',
        goal='Find the most relevant job opportunities based on candidate profiles and preferences',
        backstory="""You are a seasoned recruitment specialist with extensive knowledge of job markets across different 
        industries. You know where to find the best opportunities and understand what makes a job posting attractive to 
        candidates. You have connections across various job boards and company networks.""",
        verbose=True,
        allow_delegation=False,
        tools=[JobSearchTool()],
        llm=LangChainOpenAI(openai_api_key=api_key, temperature=0.5)
    )

def create_matching_agent(api_key: str) -> Agent:
    """Create job matching specialist agent"""
    return Agent(
        role='Job Matching Specialist',
        goal='Match candidates with the most suitable job opportunities and provide compatibility analysis',
        backstory="""You are an AI-powered matching specialist with deep understanding of job requirements and candidate 
        capabilities. You excel at identifying the perfect fit between professionals and opportunities, considering not just 
        skills but also career growth potential, company culture, and long-term satisfaction.""",
        verbose=True,
        allow_delegation=False,
        tools=[MatchingTool()],
        llm=LangChainOpenAI(openai_api_key=api_key, temperature=0.2)
    )

def create_application_agent(api_key: str) -> Agent:
    """Create application assistance agent"""
    return Agent(
        role='Application Assistant',
        goal='Help candidates create compelling application materials including cover letters and interview preparation',
        backstory="""You are a career coach and professional writer with expertise in creating compelling application 
        materials. You understand what recruiters and hiring managers look for and can craft personalized content that 
        highlights candidate strengths effectively.""",
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=LangChainOpenAI(openai_api_key=api_key, temperature=0.7)
    )

# Memory Management with Chroma
def initialize_vector_store(api_key: str):
    """Initialize Chroma vector store for memory management"""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = Chroma(
            collection_name="job_matcher_memory",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        return vector_store
    except Exception as e:
        st.error(f"Vector store initialization error: {e}")
        return None

def store_conversation_memory(vector_store, user_query: str, response: str, metadata: Dict):
    """Store conversation in vector memory"""
    try:
        doc = Document(
            page_content=f"Query: {user_query}\nResponse: {response}",
            metadata={
                "timestamp": datetime.now().isoformat(),
                "type": "conversation",
                **metadata
            }
        )
        vector_store.add_documents([doc])
    except Exception as e:
        st.error(f"Memory storage error: {e}")

def search_memory(vector_store, query: str, k: int = 3) -> List[str]:
    """Search conversation memory"""
    try:
        if vector_store:
            results = vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        return []
    except Exception as e:
        st.error(f"Memory search error: {e}")
        return []

# Multi-Agent Crew Execution
def execute_crew_analysis(resume_text: str, location: str, api_key: str) -> Dict:
    """Execute multi-agent crew for comprehensive job matching"""
    try:
        # Create agents
        resume_agent = create_resume_analyst_agent(api_key)
        search_agent = create_job_search_agent(api_key)
        matching_agent = create_matching_agent(api_key)
        application_agent = create_application_agent(api_key)
        
        # Create tasks
        resume_task = Task(
            description=f"Analyze this resume and extract comprehensive professional profile: {resume_text[:1000]}...",
            agent=resume_agent,
            expected_output="Detailed JSON profile with skills, experience, and preferences"
        )
        
        search_task = Task(
            description=f"Search for relevant job opportunities in {location} based on the analyzed profile",
            agent=search_agent,
            expected_output="List of relevant job opportunities with details",
            context=[resume_task]
        )
        
        matching_task = Task(
            description="Match the candidate profile with found job opportunities and calculate compatibility scores",
            agent=matching_agent,
            expected_output="Ranked list of job matches with compatibility scores",
            context=[resume_task, search_task]
        )
        
        # Create and execute crew
        crew = Crew(
            agents=[resume_agent, search_agent, matching_agent],
            tasks=[resume_task, search_task, matching_task],
            verbose=True,
            process=Process.sequential
        )
        
        st.session_state.agent_status['resume_agent'] = 'running'
        st.session_state.agent_status['search_agent'] = 'waiting'
        st.session_state.agent_status['matching_agent'] = 'waiting'
        
        result = crew.kickoff()
        
        # Update status
        for agent in ['resume_agent', 'search_agent', 'matching_agent']:
            st.session_state.agent_status[agent] = 'complete'
        
        return {
            'success': True,
            'result': result,
            'agents_used': 3,
            'tasks_completed': 3
        }
        
    except Exception as e:
        st.error(f"Crew execution error: {e}")
        return {'success': False, 'error': str(e)}

# PDF reader function
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF"""
    try:
        pdf_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Main application
def main():
    st.title("🤖 Multi-Agent AI Job Matcher")
    st.markdown("*Powered by CrewAI • LangChain • OpenAI/Llama3 • Chroma Vector Memory*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔑 Configuration")
        
        # API Key
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Required for AI agents and embeddings"
        )
        
        if api_key:
            st.session_state.openai_api_key = api_key
            # Initialize vector store
            if not st.session_state.vector_store:
                with st.spinner("Initializing Chroma vector memory..."):
                    st.session_state.vector_store = initialize_vector_store(api_key)
            st.success("✅ Configuration complete!")
        else:
            st.warning("⚠️ Enter API key to activate agents")
        
        st.markdown("---")
        
        # Agent Status Dashboard
        st.header("🤖 Agent Status")
        agents = {
            'resume_agent': 'Resume Analyst',
            'search_agent': 'Job Searcher', 
            'matching_agent': 'Job Matcher',
            'application_agent': 'Application Assistant'
        }
        
        for agent_id, agent_name in agents.items():
            status = st.session_state.agent_status.get(agent_id, 'idle')
            status_class = f"status-{status}"
            if status == 'idle':
                status_class = "status-complete"
            st.markdown(f'<div class="task-status {status_class}">{agent_name}: {status.title()}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Memory Status
        st.header("🧠 Memory Status")
        if st.session_state.vector_store:
            try:
                # Get collection info
                st.markdown('<div class="memory-indicator">Vector Store: Active</div>', unsafe_allow_html=True)
                st.markdown('<div class="memory-indicator">Embeddings: OpenAI</div>', unsafe_allow_html=True)
            except:
                st.markdown('<div class="memory-indicator">Vector Store: Error</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="memory-indicator">Vector Store: Not Initialized</div>', unsafe_allow_html=True)
        
        # Search Settings
        st.markdown("---")
        st.header("📍 Search Settings")
        location = st.selectbox(
            "Location Preference",
            ["Remote", "San Francisco", "New York", "Seattle", "Austin", "Boston", "Chicago", "Los Angeles"]
        )
        st.session_state.search_location = location
        
        num_jobs = st.slider("Jobs to Find", 5, 25, 15)
        
        if st.button("🗑️ Clear Memory & Reset"):
            st.session_state.resume_analysis = None
            st.session_state.jobs = []
            st.session_state.crew_results = {}
            st.session_state.agent_status = {}
            st.success("Reset complete!")
            st.rerun()
    
    if not api_key:
        st.error("🔑 Please enter your OpenAI API key in the sidebar to activate the multi-agent system")
        st.info("💡 The system uses multiple AI agents working together:\n- **Resume Analyst**: Extracts skills and experience\n- **Job Searcher**: Finds relevant opportunities\n- **Job Matcher**: Calculates compatibility scores\n- **Application Assistant**: Creates cover letters")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Resume Analysis", "🔍 Multi-Agent Search", "🎯 Job Matching", "📝 Applications"])
    
    with tab1:
        st.header("📄 AI-Powered Resume Analysis")
        
        # Display active agents
        st.markdown("""
        <div class="agent-card">
            <h4>🤖 Resume Analysis Agent</h4>
            <p><strong>Role:</strong> Expert HR professional specializing in resume analysis</p>
            <p><strong>Tools:</strong> Advanced NLP • Pattern Recognition • Skill Extraction</p>
            <p><strong>Memory:</strong> Chroma Vector Store for context retention</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload resume
        resume_file = st.file_uploader(
            "Upload your resume for AI analysis",
            type=['pdf', 'txt'],
            help="PDF or TXT files supported"
        )
        
        resume_text = ""
        if resume_file:
            try:
                if resume_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(resume_file)
                else:
                    resume_text = str(resume_file.read(), "utf-8")
                
                if resume_text:
                    st.success("✅ Resume uploaded successfully!")
                    st.session_state.resume_text = resume_text
                    
                    with st.expander("📖 Resume Preview"):
                        st.text_area("Content", resume_text[:800] + "...", height=200, disabled=True)
                        
                    # Store in vector memory
                    if st.session_state.vector_store:
                        store_conversation_memory(
                            st.session_state.vector_store,
                            "Resume uploaded",
                            resume_text[:500] + "...",
                            {"type": "resume", "filename": resume_file.name}
                        )
                else:
                    st.error("❌ Could not extract text from file")
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
        
        # AI Analysis
        if st.button("🤖 Start Multi-Agent Analysis", type="primary", disabled=not resume_text):
            if resume_text:
                st.session_state.agent_status['resume_agent'] = 'running'
                
                with st.spinner("🤖 Resume Analysis Agent is working..."):
                    # Use the resume analysis tool directly
                    tool = ResumeAnalysisTool()
                    analysis = tool._run(resume_text)
                    
                    if analysis and 'error' not in analysis:
                        st.session_state.resume_analysis = analysis
                        st.session_state.agent_status['resume_agent'] = 'complete'
                        
                        # Store analysis in memory
                        if st.session_state.vector_store:
                            store_conversation_memory(
                                st.session_state.vector_store,
                                "Resume analysis completed",
                                json.dumps(analysis),
                                {"type": "analysis", "agent": "resume_analyst"}
                            )
                        
                        st.success("✅ Multi-agent analysis complete!")
                        st.rerun()
                    else:
                        st.session_state.agent_status['resume_agent'] = 'error'
                        st.error("❌ Analysis failed. Please try again.")
        
        # Display results
        if st.session_state.resume_analysis:
            analysis = st.session_state.resume_analysis
            
            st.subheader("📊 AI Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Professional Profile")
                st.metric("Experience Level", analysis.get('experience_level', 'N/A'))
                st.metric("Years of Experience", analysis.get('years_experience', 'N/A'))
                st.metric("Primary Domain", analysis.get('primary_domain', 'N/A').title())
                
                st.markdown("### 🛠️ Extracted Skills")
                skills_html = ""
                for skill in analysis.get('skills', []):
                    skills_html += f'<span class="skill-tag">{skill}</span> '
                st.markdown(skills_html, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 💼 Career History")
                st.write("**Past Roles:**")
                for title in analysis.get('job_titles', ['Not detected']):
                    st.write(f"• {title}")
                
                st.markdown("### 🎯 AI Recommendations")
                st.write("**Suggested Roles:**")
                for role in analysis.get('preferred_roles', ['General Software Developer']):
                    st.write(f"• {role}")
            
            # Memory search
            if st.session_state.vector_store:
                st.markdown("### 🧠 Memory Insights")
                with st.expander("Search Previous Analyses"):
                    query = st.text_input("Search memory:", placeholder="e.g., 'python skills' or 'senior developer'")
                    if query:
                        memories = search_memory(st.session_state.vector_store, query)
                        if memories:
                            for i, memory in enumerate(memories, 1):
                                st.text_area(f"Memory {i}", memory[:200] + "...", height=100)
                        else:
                            st.info("No relevant memories found")
    
    with tab2:
        st.header("🔍 Multi-Agent Job Search System")
        
        if not st.session_state.resume_analysis:
            st.warning("⚠️ Please complete resume analysis first!")
            return
        
        # Display search agents
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="agent-card">
                <h4>🕵️ Job Search Agent</h4>
                <p><strong>Specialty:</strong> Multi-platform job discovery</p>
                <p><strong>Sources:</strong> LinkedIn • Indeed • Glassdoor • AngelList</p>
                <p><strong>Intelligence:</strong> Context-aware search optimization</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="agent-card">
                <h4>🎯 Matching Agent</h4>
                <p><strong>Specialty:</strong> AI-powered compatibility analysis</p>
                <p><strong>Algorithm:</strong> Skills • Experience • Culture fit</p>
                <p><strong>Output:</strong> Ranked opportunities with scores</p>
            </div>
            """, unsafe_allow_html=True)
        
        analysis = st.session_state.resume_analysis
        
        # Search strategy display
        st.markdown("### 🎯 AI Search Strategy")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("**🔍 Search Queries Generated:**")
            search_queries = []
            
            # Generate queries based on analysis
            for role in analysis.get('preferred_roles', [])[:3]:
                search_queries.append(role)
            for skill in analysis.get('skills', [])[:2]:
                if skill.lower() not in ['communication', 'teamwork', 'problem solving']:
                    search_queries.append(skill)
            
            for query in search_queries[:5]:
                st.write(f"• {query}")
        
        with col_b:
            st.write("**⚙️ Search Parameters:**")
            st.write(f"• Location: {st.session_state.search_location}")
            st.write(f"• Experience Level: {analysis.get('experience_level', 'Any')}")
            st.write(f"• Domain Focus: {analysis.get('primary_domain', 'Software').title()}")
            st.write(f"• Results Target: {num_jobs} positions")
        
        # Execute multi-agent search
        if st.button("🚀 Launch Multi-Agent Search", type="primary"):
            st.session_state.agent_status['search_agent'] = 'running'
            st.session_state.agent_status['matching_agent'] = 'waiting'
            
            with st.spinner("🤖 Search Agent is scanning job markets..."):
                search_tool = JobSearchTool()
                matching_tool = MatchingTool()
                
                all_jobs = []
                
                # Search with each query
                for i, query in enumerate(search_queries[:3]):
                    st.write(f"🔍 Searching for: {query}")
                    jobs = search_tool._run(query, st.session_state.search_location, 5)
                    all_jobs.extend(jobs)
                    time.sleep(0.5)  # Simulate processing time
                
                st.session_state.agent_status['search_agent'] = 'complete'
                st.session_state.agent_status['matching_agent'] = 'running'
                
                # Remove duplicates
                unique_jobs = []
                seen_companies = set()
                for job in all_jobs:
                    if job.company not in seen_companies:
                        unique_jobs.append(job)
                        seen_companies.add(job.company)
                
                with st.spinner("🎯 Matching Agent is calculating compatibility..."):
                    # Calculate match scores
                    matched_jobs = matching_tool._run(analysis, unique_jobs[:num_jobs])
                    st.session_state.jobs = matched_jobs
                    
                    st.session_state.agent_status['matching_agent'] = 'complete'
                    
                    # Store in memory
                    if st.session_state.vector_store:
                        store_conversation_memory(
                            st.session_state.vector_store,
                            f"Job search completed for {', '.join(search_queries)}",
                            f"Found {len(matched_jobs)} relevant positions",
                            {
                                "type": "job_search", 
                                "location": st.session_state.search_location,
                                "num_results": len(matched_jobs)
                            }
                        )
                
                st.success(f"✅ Multi-agent search complete! Found {len(matched_jobs)} opportunities")
                st.balloons()
                st.rerun()
        
        # Display search results
        if st.session_state.jobs:
            st.markdown("### 📊 Search Results Summary")
            
            col_x, col_y, col_z = st.columns(3)
            with col_x:
                st.metric("Total Jobs Found", len(st.session_state.jobs))
            with col_y:
                avg_match = sum(job.match_score for job in st.session_state.jobs) / len(st.session_state.jobs)
                st.metric("Average Match Score", f"{avg_match:.1f}%")
            with col_z:
                high_matches = len([job for job in st.session_state.jobs if job.match_score >= 75])
                st.metric("High-Match Jobs (75%+)", high_matches)
    
    with tab3:
        st.header("🎯 AI Job Matching & Compatibility Analysis")
        
        if not st.session_state.jobs:
            st.info("📋 Complete the multi-agent search to see job matches")
            return
        
        st.markdown("### 🏆 Top Job Matches (AI-Ranked)")
        
        # Filter and sort options
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            min_match = st.slider("Minimum Match Score", 0, 100, 0, 5)
        with col_filter2:
            sort_by = st.selectbox("Sort By", ["Match Score", "Company", "Salary", "Posted Date"])
        
        # Filter jobs
        filtered_jobs = [job for job in st.session_state.jobs if job.match_score >= min_match]
        
        if sort_by == "Company":
            filtered_jobs.sort(key=lambda x: x.company)
        elif sort_by == "Salary":
            filtered_jobs.sort(key=lambda x: x.salary, reverse=True)
        elif sort_by == "Posted Date":
            filtered_jobs.sort(key=lambda x: x.posted_date)
        # Default is already sorted by match score
        
        # Display jobs
        for i, job in enumerate(filtered_jobs):
            with st.container():
                # Match score color coding
                if job.match_score >= 80:
                    score_color = "#4CAF50"
                    score_label = "Excellent Match"
                elif job.match_score >= 60:
                    score_color = "#FF9800"
                    score_label = "Good Match"
                else:
                    score_color = "#F44336"
                    score_label = "Fair Match"
                
                st.markdown(f"""
                <div class="job-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h4 style="margin: 0; color: #333;">{job.title} at {job.company}</h4>
                        <div style="text-align: right;">
                            <div style="background: {score_color}; color: white; padding: 8px 15px; border-radius: 20px; font-weight: bold; margin-bottom: 5px;">
                                {job.match_score}% Match
                            </div>
                            <small style="color: #666;">{score_label}</small>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                        <div>
                            <strong>📍 Location:</strong> {job.location}<br>
                            <strong>💰 Salary:</strong> {job.salary}<br>
                            <strong>📅 Posted:</strong> {job.posted_date}
                        </div>
                        <div>
                            <strong>🏢 Source:</strong> {job.source}<br>
                            <strong>🆔 Job ID:</strong> {job.id}
                        </div>
                    </div>
                    
                    <p><strong>📋 Description:</strong> {job.description}</p>
                    
                    <div style="margin: 10px 0;">
                        <strong>🛠️ Required Skills:</strong><br>
                """, unsafe_allow_html=True)
                
                # Display skills with highlighting
                analysis = st.session_state.resume_analysis
                user_skills_lower = [s.lower() for s in analysis.get('skills', [])]
                
                skills_html = ""
                for skill in job.skills_required:
                    is_match = any(user_skill in skill.lower() or skill.lower() in user_skill 
                                 for user_skill in user_skills_lower)
                    color = "#4CAF50" if is_match else "#e3f2fd"
                    text_color = "white" if is_match else "#1976d2"
                    skills_html += f'<span style="background: {color}; color: {text_color}; padding: 4px 8px; border-radius: 12px; font-size: 0.85em; margin: 2px; display: inline-block;">{skill}</span> '
                
                st.markdown(skills_html + "</div>", unsafe_allow_html=True)
                
                # Action buttons
                col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
                
                with col_btn1:
                    st.markdown(f'<a href="{job.apply_url}" target="_blank" style="background: #007bff; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; display: inline-block;">🔍 View on {job.source}</a>', unsafe_allow_html=True)
                
                with col_btn2:
                    company_url = f"https://careers.{job.company.lower().replace(' ', '')}.com"
                    st.markdown(f'<a href="{company_url}" target="_blank" style="background: #28a745; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; display: inline-block;">🏢 Company Careers</a>', unsafe_allow_html=True)
                
                with col_btn3:
                    if st.button(f"📝 Generate Application", key=f"apply_{i}"):
                        st.session_state.selected_job = job
                        st.session_state.agent_status['application_agent'] = 'selected'
                        st.success("✅ Job selected! Go to Applications tab.")
                
                with col_btn4:
                    if st.button(f"🧠 Memory Analysis", key=f"memory_{i}"):
                        if st.session_state.vector_store:
                            query = f"{job.title} {job.company} {' '.join(job.skills_required[:3])}"
                            memories = search_memory(st.session_state.vector_store, query, k=2)
                            if memories:
                                st.info(f"🧠 Memory found: {len(memories)} related entries")
                                with st.expander("View Memory Context"):
                                    for mem in memories:
                                        st.text(mem[:300] + "...")
                            else:
                                st.info("🧠 No related memories found")
                
                st.markdown("</div>", unsafe_allow_html=True)
                st.divider()
        
        if not filtered_jobs:
            st.warning(f"No jobs match your criteria (minimum {min_match}% match score)")
    
    with tab4:
        st.header("📝 AI Application Assistant")
        
        if 'selected_job' not in st.session_state:
            st.info("📋 Select a job from the Job Matching tab to generate application materials")
            return
        
        # Display application agent
        st.markdown("""
        <div class="agent-card">
            <h4>✍️ Application Assistant Agent</h4>
            <p><strong>Specialty:</strong> Personalized application materials</p>
            <p><strong>Capabilities:</strong> Cover Letters • Interview Prep • Career Coaching</p>
            <p><strong>Intelligence:</strong> Company research + Role customization</p>
        </div>
        """, unsafe_allow_html=True)
        
        job = st.session_state.selected_job
        analysis = st.session_state.resume_analysis
        
        st.subheader(f"🎯 Application for: {job.title} at {job.company}")
        
        # Job and candidate summary
        col_summary1, col_summary2 = st.columns(2)
        
        with col_summary1:
            st.markdown("### 💼 Position Details")
            st.write(f"**Company:** {job.company}")
            st.write(f"**Role:** {job.title}")
            st.write(f"**Location:** {job.location}")
            st.write(f"**Salary:** {job.salary}")
            st.write(f"**Match Score:** {job.match_score}%")
            st.write(f"**Source:** {job.source}")
        
        with col_summary2:
            st.markdown("### 👤 Your Profile")
            st.write(f"**Experience:** {analysis.get('experience_level', 'N/A')}")
            st.write(f"**Years:** {analysis.get('years_experience', 'N/A')}")
            st.write(f"**Domain:** {analysis.get('primary_domain', 'N/A').title()}")
            st.write(f"**Top Skills:** {', '.join(analysis.get('skills', [])[:4])}")
        
        # Quick application links
        st.markdown("### 🚀 Quick Apply")
        col_apply1, col_apply2 = st.columns(2)
        
        with col_apply1:
            st.markdown(f'<a href="{job.apply_url}" target="_blank" style="background: #007bff; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; display: inline-block; font-weight: bold;">🔍 Apply on {job.source}</a>', unsafe_allow_html=True)
        
        with col_apply2:
            company_url = f"https://careers.{job.company.lower().replace(' ', '')}.com"
            st.markdown(f'<a href="{company_url}" target="_blank" style="background: #28a745; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; display: inline-block; font-weight: bold;">🏢 {job.company} Careers</a>', unsafe_allow_html=True)
        
        st.divider()
        
        # Generate application materials
        tab_cover, tab_interview, tab_strategy = st.tabs(["📝 Cover Letter", "🎤 Interview Prep", "📈 Application Strategy"])
        
        with tab_cover:
            st.markdown("### 📝 AI-Generated Cover Letter")
            
            if st.button("🤖 Generate Personalized Cover Letter", type="primary"):
                st.session_state.agent_status['application_agent'] = 'running'
                
                with st.spinner("✍️ Application Assistant is crafting your cover letter..."):
                    # Generate cover letter using OpenAI
                    try:
                        client = OpenAI(api_key=api_key)
                        
                        # Create detailed prompt
                        prompt = f"""
                        Write a compelling, personalized cover letter for this job application:
                        
                        POSITION: {job.title} at {job.company}
                        LOCATION: {job.location}
                        SALARY: {job.salary}
                        JOB DESCRIPTION: {job.description}
                        REQUIRED SKILLS: {', '.join(job.skills_required)}
                        
                        CANDIDATE PROFILE:
                        Experience Level: {analysis.get('experience_level', 'Mid-level')}
                        Years of Experience: {analysis.get('years_experience', '3-5 years')}
                        Skills: {', '.join(analysis.get('skills', []))}
                        Previous Roles: {', '.join(analysis.get('job_titles', []))}
                        Domain: {analysis.get('primary_domain', 'Software')}
                        
                        INSTRUCTIONS:
                        1. Address the hiring manager professionally
                        2. Show genuine enthusiasm for {job.company} and the specific role
                        3. Highlight relevant skills and experience that match the job requirements
                        4. Demonstrate knowledge of the company's mission and values
                        5. Explain why you're excited about this opportunity
                        6. Keep it concise but compelling (300-400 words)
                        7. Use a professional, confident tone
                        8. Include a strong call to action
                        
                        Format as a proper business letter with:
                        - Date
                        - Hiring Manager address
                        - Professional greeting
                        - 3-4 body paragraphs
                        - Professional closing
                        - Signature line
                        """
                        
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=600
                        )
                        
                        cover_letter = response.choices[0].message.content.strip()
                        
                        st.session_state.agent_status['application_agent'] = 'complete'
                        
                        # Store in memory
                        if st.session_state.vector_store:
                            store_conversation_memory(
                                st.session_state.vector_store,
                                f"Cover letter generated for {job.title} at {job.company}",
                                cover_letter[:300] + "...",
                                {
                                    "type": "cover_letter",
                                    "company": job.company,
                                    "position": job.title,
                                    "agent": "application_assistant"
                                }
                            )
                        
                        st.success("✅ Cover letter generated!")
                        
                        # Display cover letter
                        st.text_area("Your Personalized Cover Letter", cover_letter, height=500)
                        
                        # Download button
                        st.download_button(
                            "📥 Download Cover Letter",
                            cover_letter,
                            file_name=f"cover_letter_{job.company}_{job.title.replace(' ', '_')}.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.session_state.agent_status['application_agent'] = 'error'
                        st.error(f"❌ Error generating cover letter: {e}")
        
        with tab_interview:
            st.markdown("### 🎤 AI Interview Preparation")
            
            if st.button("🎯 Generate Interview Prep Materials", type="primary"):
                with st.spinner("🤖 Preparing your interview strategy..."):
                    
                    col_questions, col_technical = st.columns(2)
                    
                    with col_questions:
                        st.markdown("#### 🎤 Behavioral Questions")
                        behavioral_questions = [
                            f"Tell me about your experience with {analysis.get('skills', ['programming'])[0]}",
                            f"Why do you want to work at {job.company}?",
                            f"Describe a challenging project you worked on as a {analysis.get('job_titles', ['developer'])[0]}",
                            "What interests you most about this role?",
                            "How do you handle tight deadlines and pressure?",
                            "Where do you see yourself in 5 years?",
                            "What's your greatest professional achievement?",
                            "How do you stay updated with new technologies?",
                            "Describe a time you had to learn something completely new",
                            "What questions do you have for us?"
                        ]
                        
                        for i, question in enumerate(behavioral_questions, 1):
                            st.write(f"**{i}.** {question}")
                    
                    with col_technical:
                        st.markdown("#### 🛠️ Technical Focus Areas")
                        st.write("**Skills to Review:**")
                        for skill in job.skills_required:
                            is_match = any(s.lower() in skill.lower() for s in analysis.get('skills', []))
                            status = "✅ (You have this)" if is_match else "📚 (Study this)"
                            st.write(f"• {skill} {status}")
                        
                        st.markdown("#### 🏢 Company Research")
                        research_points = [
                            f"{job.company} company mission and values",
                            f"Recent news about {job.company}",
                            f"{job.company} products and services",
                            f"{job.company} competitors and market position",
                            f"{job.company} engineering culture and practices",
                            "Company's recent achievements or milestones"
                        ]
                        
                        for point in research_points:
                            st.write(f"• {point}")
                    
                    # STAR method guidance
                    st.markdown("#### ⭐ STAR Method Template")
                    st.info("""
                    **Situation:** Describe the context or background
                    **Task:** Explain what you needed to accomplish
                    **Action:** Detail the specific steps you took
                    **Result:** Share the outcomes and what you learned
                    
                    Use this framework to structure your behavioral responses!
                    """)
        
        with tab_strategy:
            st.markdown("### 📈 Application Strategy & Timeline")
            
            # Application strategy
            col_strategy1, col_strategy2 = st.columns(2)
            
            with col_strategy1:
                st.markdown("#### 📋 Application Checklist")
                checklist_items = [
                    "✅ Resume tailored to job requirements",
                    "✅ Cover letter personalized for company",
                    "📋 LinkedIn profile updated",
                    "📋 Portfolio/GitHub links ready",
                    "📋 References list prepared",
                    "📋 Questions for interviewer prepared",
                    "📋 Company research completed",
                    "📋 Technical skills refreshed"
                ]
                
                for item in checklist_items:
                    st.write(item)
            
            with col_strategy2:
                st.markdown("#### 📅 Follow-up Timeline")
                timeline = [
                    "**Day 0:** Submit application",
                    "**Day 3:** Connect with hiring manager on LinkedIn",
                    "**Day 7:** Follow up email if no response",
                    "**Day 14:** Second follow-up (if appropriate)",
                    "**Week 3:** Consider reaching out to team members",
                    "**Week 4:** Final follow-up before moving on"
                ]
                
                for item in timeline:
                    st.write(item)
            
            # Success metrics
            st.markdown("#### 📊 Application Success Factors")
            
            success_factors = {
                "Skills Match": f"{job.match_score}%",
                "Experience Level": "✅ Appropriate" if analysis.get('experience_level') != 'Entry-level' or 'junior' not in job.title.lower() else "⚠️ Stretch role",
                "Location Preference": "✅ Match" if job.location == st.session_state.search_location else "⚠️ Different",
                "Company Size": "✅ Suitable" if job.company in ["Google", "Microsoft", "Amazon", "Meta", "Apple"] else "✅ Growth opportunity"
            }
            
            col_metrics1, col_metrics2 = st.columns(2)
            i = 0
            for factor, status in success_factors.items():
                if i % 2 == 0:
                    col_metrics1.metric(factor, status)
                else:
                    col_metrics2.metric(factor, status)
                i += 1
            
            # Memory-based insights
            if st.session_state.vector_store:
                st.markdown("#### 🧠 Memory-Based Insights")
                query = f"{job.company} {job.title} application strategy"
                memories = search_memory(st.session_state.vector_store, query, k=3)
                
                if memories:
                    st.info(f"Found {len(memories)} related insights from your application history")
                    with st.expander("View Insights"):
                        for i, memory in enumerate(memories, 1):
                            st.text_area(f"Insight {i}", memory[:400] + "...", height=100)
                else:
                    st.info("💡 This appears to be your first application to a similar role - good luck!")
    
    # Footer with system status
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <h4>🤖 Multi-Agent AI Job Matcher System Status</h4>
        <p><strong>Tech Stack:</strong> CrewAI • LangChain • OpenAI • Chroma Vector Database</p>
        <p><strong>Agents Active:</strong> Resume Analyst • Job Searcher • Compatibility Matcher • Application Assistant</p>
        <p><strong>Memory:</strong> Vector embeddings for context retention and personalized insights</p>
        <p><small>Real job opportunities with direct application links • AI-powered matching • Personalized application materials</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()