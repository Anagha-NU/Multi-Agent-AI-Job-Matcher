# Multi-Agent AI Job Matcher

🤖 An intelligent multi-agent system for analyzing resumes, discovering relevant jobs, matching candidates with top opportunities, and generating personalized application materials.

---

## Overview

This project leverages a multi-agent AI architecture using CrewAI, LangChain, and OpenAI to provide a comprehensive job matching platform.

Key features include:
- AI-powered resume analysis to extract skills, experience, roles, and preferences.
- Multi-platform job search across LinkedIn, Indeed, Glassdoor, AngelList, and more.
- AI-driven job matching with compatibility scoring based on skills, experience, and location.
- Application assistant to generate personalized cover letters, interview prep, and strategy.
- Contextual memory and conversation history stored using Chroma vector database.
- Built with Streamlit for an interactive web UI.

---

## Architecture & Technologies

- **CrewAI** for orchestrating multiple specialist agents (Resume Analyst, Job Searcher, Job Matcher, Application Assistant)
- **LangChain** to integrate AI tools and manage prompts/agents
- **OpenAI GPT-3.5-turbo / Llama3** as the language models powering NLP tasks
- **Chroma Vector Database** for memory management with OpenAI embeddings
- **Streamlit** for creating an easy-to-use web interface
- **PyPDF2** for PDF resume text extraction

---

## Features

### Resume Analysis Agent
- Extracts skills, job titles, experience level, and career preferences from uploaded resumes (PDF or TXT).
- Uses NLP and pattern matching to identify top technical skills and suggest roles.

### Job Search Agent
- Searches relevant jobs based on analyzed user profile.
- Generates realistic job listings with company, salary ranges, descriptions, and required skills.

### Job Matching Agent
- Scores job matches based on compatibility with candidate's profile.
- Considers skill overlap, experience level, and location preferences.

### Application Assistant Agent
- Creates custom cover letters tailored to selected job listings.
- Provides interview preparation questions and application strategy guidance.
- Stores insights and interaction history in vector memory for better personalization.

---

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (for GPT models and embeddings)
- Install dependencies:

pip install -r requirements.txt

text

### Run the Application

streamlit run main.py

text

### Usage

1. Enter your OpenAI API key in the sidebar.
2. Upload your resume (PDF or TXT).
3. Start the multi-agent resume analysis.
4. Launch job search and matching agents to find relevant jobs.
5. Browse AI-ranked job matches with compatibility scores.
6. Select a job to create personalized application materials and interview prep.

---

## Project Structure

- `main.py` — Streamlit front-end and app orchestration.
- `agents.py` — CrewAI agents definitions and tool implementations.
- `tools.py` — Resume analysis, job search, and matching tools.
- `memory.py` — Vector store initialization and search utilities.
- `utils.py` — Helper functions for PDF extraction, text processing.

---

## Customization

- Supported job locations can be modified in the sidebar dropdown.
- Number of jobs to retrieve is adjustable via slider.
- Extend skills and job title extraction regex patterns within `ResumeAnalysisTool`.
- Add new job sources or customize search behavior in `JobSearchTool`.

---

