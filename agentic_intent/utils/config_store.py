import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class ConfigStore:
    """Central configuration store for system parameters"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, "system_config.json")
        self.config_path = config_path
        self._ensure_config_exists()
    
    def _ensure_config_exists(self):
        """Create default config if it doesn't exist"""
        if not os.path.exists(self.config_path):
            default_config = {
                "search_params": {
                    "funding_max_results": 5,
                    "partnerships_max_results": 5,
                    "news_max_results": 5
                },
                "confidence_thresholds": {
                    "funding_min_confidence": 0.3,
                    "news_min_confidence": 0.3,
                    "event_confidence_threshold": 0.5
                },
                "source_preferences": {
                    "trusted_sources": [],
                    "blocked_sources": []
                },
                "custom_search_queries": {},
                "prompts": {
                    "funding_aggregation": {
                        "version": 1,
                        "template": """
You are a financial data extraction system. Your task is to identify and group funding events for {company}.

INPUT:
{events}

RULES:
1. Group news articles that describe the SAME funding round/event
2. Each unique funding event = 1 output object
3. Use ONLY information from the provided articles
4. If a detail is not mentioned, set it to null (not "None" string, not "Unknown")
5. Every article ID must appear in exactly ONE group (no duplicates, no orphans)

OUTPUT FORMAT - Return a JSON list of objects:
[
  {{
    "event_title": "Brief descriptive of the event",
    "event_confidence": 0.0 to 1.0,
    "source": "single most authoritative source name from the grouped articles",
    "supporting_ids": ["id1", "id2"],
    "date": "extracted date or null",
    "date_confidence": 0.0 to 1.0,
    "investor": "investor name(s) or null",
    "investor_confidence": 0.0 to 1.0,
    "amount": "funding amount with currency or null",
    "amount_confidence": 0.0 to 1.0
  }}
]

Return ONLY the JSON array, nothing else.
"""
                    },
                    "news_aggregation": {
                        "version": 1,
                        "template": """
You are a news clustering system.

Group news articles that describe the same real-world event related to this company: {company}.

INPUT:
{events}

TASKS:

1. COMPANY FILTERING
- Only consider articles that are primarily about the specified company
- The company must be a central subject, not just mentioned in passing

2. CLUSTERING
- Group articles referring to the same event
- Same event = same incident/announcement/development
- Do NOT mix unrelated news

3. OUTPUT PER CLUSTER
Return one object per event:
- event_title: brief description of the event
- supporting_ids: all article IDs in the cluster
- source: BEST single source from within the cluster
- event_confidence: 0.8-1.0 strong, 0.5-0.8 partial, <0.5 weak

Return ONLY valid JSON list:
[
  {{
    "event_title": "",
    "event_confidence": 0.0,
    "source": "",
    "supporting_ids": []
  }}
]
"""
                    },
                    "funding_extraction": {
                        "version": 1,
                        "template": """
You are an information extraction system.

Company: {company}

Input:
{events}

For each element:
- If NOT related to the company → return: id: null
- If related → return:
  id: {{"date": "dd/mm/yyyy or None", "investor": "name or None", "amount": "value or None"}}

Return ONLY JSON: {{"0": {{...}}, "1": null}}
"""
                    }
                },
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "updated_by": "system_init"
                }
            }
            self.save(default_config)
    
    def load(self) -> Dict:
        """Load configuration"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def save(self, config: Dict):
        """Save configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get(self, key_path: str, default=None):
        """
        Get a config value using dot notation.
        Example: config.get("search_params.funding_max_results")
        """
        config = self.load()
        keys = key_path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Set a config value using dot notation.
        Example: config.set("search_params.funding_max_results", 10)
        """
        config = self.load()
        keys = key_path.split('.')
        target = config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
        
        config["metadata"]["last_updated"] = datetime.now().isoformat()
        config["metadata"]["updated_by"] = "orchestrator"
        self.save(config)
    
    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """Get a prompt template by name"""
        prompt_data = self.get(f"prompts.{prompt_name}")
        if prompt_data:
            return prompt_data.get("template")
        return None
    
    def update_prompt(self, prompt_name: str, new_template: str):
        """Update a prompt template"""
        prompt_data = self.get(f"prompts.{prompt_name}")
        if prompt_data:
            prompt_data["version"] += 1
            prompt_data["template"] = new_template
            self.set(f"prompts.{prompt_name}", prompt_data)


# Global instance
config_store = ConfigStore()