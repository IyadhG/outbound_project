Nudge AI - Autonomous Outbound Sales Agent  Overview  This project, Nudge AI, was developed as part of the Artificial Intelligence (4IA) coursework at Esprit School of Engineering. It is an autonomous agentic system designed to solve the "B2B Pipeline Black Hole" by automating lead research, qualification, and hyper-personalized outreach.  Features  The system is divided into four specialized modules that function as a unified pipeline:1. Inject / Collect (Sensory Input): Automates the discovery and enrichment of leads by scraping real-time intent signals from LinkedIn and corporate websites.  2. Detective (Intelligence Engine): Features a multi-agent "AI Jury" (Optimist, Skeptic, and Auditor) built on LangGraph to deliberate on lead quality and map social relationships using Neo4j.  3. Writer (Execution Voice): Utilizes Tone Cloning and contextual justifications to generate personalized messages that move beyond generic templates.  4. Worker (The Orchestrator): Manages the operational execution, maintains the "Red Thread" (Correlation ID) for full audit trails, and monitors delivery status in real-time.  Tech Stack  Frontend  React.js & TypeScript: For a robust, type-safe user interface.  Tailwind CSS: For responsive, professional dashboard styling.  Backend  Python: The core language for AI logic and automation.  LangGraph (LangChain): For complex multi-agent orchestration and reasoning loops.  FastAPI: To provide high-performance, asynchronous communication between modules.  Data & Infrastructure  PostgreSQL: For transactional lead data and staging.  Neo4j: To map and query the "Spiderweb" of social and professional relationships.  Docker: For containerized deployment and environment consistency.  Directory Structure  Plaintext├── apps/
│   ├── frontend/        # React/TypeScript Dashboard
│   └── backend/         # FastAPI Gateway & API logic
├── modules/
│   ├── inject/          # Scraper & Enrichment logic
│   ├── detective/       # LangGraph agents & Neo4j queries
│   ├── writer/          # LLM Tone Cloning & Prompting
│   └── worker/          # Task orchestration & Logging
├── docker-compose.yml   # Multi-container orchestration
└── README.md            # Project documentation
```[cite: 1]

## **Getting Started**[cite: 1]
To run this project locally, ensure you have **Docker** and **Python 3.11+** installed[cite: 1].
1.  Clone the repository from GitHub[cite: 1].
2.  Configure your `.env` file with API keys for OpenAI and Neo4j[cite: 1].
3.  Execute `docker-compose up --build` to launch the full stack[cite: 1].

## **Acknowledgments**[cite: 1]
This project was developed by **The Orchestrators** in collaboration with **Addvocate AI** at **Esprit Sc
