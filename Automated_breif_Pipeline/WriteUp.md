# AI Content Pipeline Agent
AI Automation Intern Assignment – Scrollhouse

This project builds an **AI agent that automates the Content Brief → Script Pipeline** for Scrollhouse, a short-form video content company.

The agent automatically converts a **client brief into a structured internal brief and script draft**, eliminating the manual rewriting step currently done by the operations team.

---

# Problem Selected

## Content Brief → Script Pipeline

Current workflow at Scrollhouse:

1. Client fills a Google Form with their content brief
2. A team member reads the form
3. They manually rewrite it into the internal Notion brief format
4. The brief is sent to the scriptwriter

This process takes **20–30 minutes per brief** and is purely manual.

The operations manager described this task as **“soul-destroying” repetitive work**.

---

# Why I Chose This Problem

I evaluated the four problems in the scenario and selected the one with:

• High automation potential  
• Low engineering complexity  
• Immediate operational impact  

### Time Cost Analysis

From the operations manager:

- 40–60 briefs processed per month
- 20–30 minutes per brief

Average calculation:

50 briefs × 25 minutes  
= **1250 minutes per month**

= **~21 hours of manual work per month**

This task involves **reformatting and rewriting information**, which is exactly the type of work large language models perform well.

---

# Solution

I built an **AI Content Pipeline Agent** that automatically:

1. Reads a client brief
2. Converts it into Scrollhouse's internal structured brief format
3. Generates a script draft for the scriptwriter

The system removes the manual middle step entirely.

---

# System Workflow
