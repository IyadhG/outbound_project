# Agentic Intent System

An intelligent system for analyzing and clustering company funding and news events using agentic workflows with LangGraph and MCP (Model Context Protocol).

## 🎯 Overview
The system uses **LangGraph** for workflow orchestration and **FastMCP** for providing tools and services via the Model Context Protocol.

- **Analyze Company Data**: Processes information about companies from multiple sources
- **Cluster Events**: Groups related articles into cohesive events using LLMs
- **Provides Explainability**: Includes XAI (Explainable AI) components to interpret system decisions
- **Evaluates Performance**: Built-in evaluation metrics
- **MCP Tools**: Uses MCP tools to enable an orchestrator agent or an admin to change / retrieve system configurations

## 📋 Project Structure

```
agentic_intent/
├── main.py                      # Main orchestrator entry point
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── docker-compose.yml           # Docker compose setup
├── system_config.json           # System configuration & prompts
├── config/                      # Configuration management
├── evaluation/                  # Evaluation & XAI components
│   ├── evaluator.py            # System performance evaluator
│   └── xai.py                  # Explainability engine & A/B testing
├── graph/                       # LangGraph workflows
│   ├── funding_graph.py        # Funding event analysis workflow
│   └── news_graph.py           # News event clustering workflow
├── mcp_server/                  # MCP Server implementation
│   ├── mcp_server.py           # FastMCP server with tools
│   ├── dev.bat                 # Development helper script
│   └── test.py                 # Server tests
├── mcp_client/                  # MCP Client
│   └── client.py               # Client implementation
└── utils/                       # Utility modules
    ├── async_utils.py          # Async helper functions
    ├── config_store.py         # Configuration management
    └── intent_store.py         # Intent/result storage
```

## 🚀 Features

### Core Capabilities

- **Multi-Agent Workflow**: Leverages LangGraph for sequential and parallel processing
- **Funding Analysis**: Intelligently extracts and groups funding events from news articles
- **News Clustering**: Groups related news articles into coherent events
- **Web Search Integration**: Uses DuckDuckGo for data gathering
- **LLM-Powered Processing**: Integrates with OpenAI and Groq
- **Event Storage**: Uses test embeddings for symentic retrieval and storage
- **Evaluation Framework**: Built-in metrics collection and system performance evaluation
- **Explainability**: XAI engine for understanding system decisions

### Configuration

The system is configurable through the MCP tools or directly from`system_config.json`:

- **Search Parameters**: Control max results for funding, partnerships, and news searches
- **Confidence Thresholds**: Set minimum confidence levels for various event types
- **Custom Queries**: Add company-specific search queries
- **Prompts**: Customize LLM prompts for different extraction tasks

Edit `system_config.json` to customize:

```json
{
  "search_params": {
    "funding_max_results": 3,
    "news_max_results": 5
  },
  "confidence_thresholds": {
    "funding_min_confidence": 0.3,
    "news_min_confidence": 0.3
  }
}
```

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional, for containerized deployment)
- API keys for LLM providers (OpenAI or Groq)

### Local Setup

1. **Clone the repository**
   ```bash
   cd agentic_intent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Docker Setup

```bash
docker-compose up -d
```

## 📦 Dependencies

Core dependencies include:

- **LangGraph**: Workflow orchestration and state management
- **LangChain**: LLM integration (OpenAI, Groq)
- **FastMCP**: Model Context Protocol server implementation
- **DuckDuckGo Search**: Web search capabilities
- **Python-dotenv**: Environment variable management

See `requirements.txt` for the complete list.

## 🏃 Usage

### Basic Execution

```bash
python main.py
```

This will:
1. Initialize the MCP client
2. Load funding and news graphs
3. Process companies specified in the configuration
4. Generate funding and news clustering results
5. Optionally evaluate system performance

### Processing Specific Companies

Modify the `COMPANIES` list in `main.py`:

```python
COMPANIES = ["Rivian", "OpenAI"]
```

### Running the MCP Server on it's own
Inside the mcp_server folder, run
```bash
.\dev.bat
```

This starts the FastMCP server that provides tools for:
- Web search
- Financial data extraction
- Evaluation metrics
- Explainability analysis

## 📊 Graphs & Workflows

### Funding Graph
Analyzes funding-related news articles to:
- Extract funding events from online sources
- Cluster related articles into funding rounds
- Extract key details (date, investor, amount)
- Assign confidence scores

### News Graph
Clusters company news articles to:
- Identify distinct events 
- Filter company-relevant news
- Assign event confidence scores
- Track supporting sources


## 📈 Evaluation & Metrics

The system includes built-in evaluation capabilities:

- **SystemEvaluator**: Collects and analyzes performance metrics
- **ExplainabilityEngine**: Provides interpretable explanations for decisions
- **Metrics Output**: JSON-formatted metrics files for analysis

Metrics are automatically saved with timestamps.

## 🔍 Explainability (XAI)

The XAI module provides:

- Decision explanations for clustering results
- Confidence score justifications
- Source reliability assessment
- Event grouping rationale

## 🐳 Docker Deployment

The project includes Docker configuration for containerized deployment:

```bash
# Build and run
docker-compose up --build

# Or just run (uses pre-built images)
docker-compose up
```

The container:
- Uses Python 3.11-slim base image
- Installs all dependencies
- Maps `/data` volume for persistent storage
- Loads `.env` from the host

## 📝 Output

The system generates:

1. **Clustering Results**: Grouped funding and news events with confidence scores
2. **Metrics Files**: Performance and evaluation data
3. **Configuration Snapshots**: System configuration used for each run
4. **Intent Store**: Processed intents and results

## 🔧 Development

### Development Scripts

Windows development helper:
```bash
mcp_server/dev.bat
```

## 📚 Key Components

### Intent Store
Manages storage and retrieval of processed intents and results.

### Config Store
Handles configuration loading, validation, and updates.


## 🚨 Error Handling

The system includes graceful error handling:
- Optional evaluation module (runs without if unavailable)
- Robust MCP client connection management
- Async exception handling

## 📞 Support

For issues or questions, please open an issue in the project repository.

## 🗺️ Roadmap

Future enhancements:
- Scraping online articles
- Job hiring signals
- Integrate APIs to verify / enrich events

---

**Last Updated**: May 2, 2026