"""
Detective service configuration — all configurable via environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -------------------------------------------------------------------------
    # Networking
    # -------------------------------------------------------------------------
    REDIS_URL: str = "redis://redis:6379"
    WORKER_URL: str = "http://api:8000"
    WRITER_URL: str = "http://writer:8003"
    PORT: int = 8002

    # -------------------------------------------------------------------------
    # LLM API keys
    # -------------------------------------------------------------------------
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_TEMPERATURE: float = 0.1

    # -------------------------------------------------------------------------
    # ICP configuration — client sets these for their use case
    # -------------------------------------------------------------------------
    ICP_CONFIG_PATH: str = "icp_config.json"

    # -------------------------------------------------------------------------
    # Scoring — client configurable thresholds
    # -------------------------------------------------------------------------
    QUALIFICATION_THRESHOLD: float = 0.6  # min final score to forward to Writer
    AUTO_FORWARD_TO_WRITER: bool = True   # if True, qualified leads go to Writer

    # -------------------------------------------------------------------------
    # Sender company details — the client configures these
    # -------------------------------------------------------------------------
    SENDER_COMPANY_NAME: str = "Your Company"
    SENDER_ELEVATOR_PITCH: str = ""
    SENDER_VALUE_PROPS: str = ""          # comma-separated
    OFFER_NAME: str = "Product Demo"
    OFFER_SOLUTION_SUMMARY: str = ""
    OFFER_CTA: str = "Book a 15 min call"
    OFFER_PAIN_POINTS: str = ""           # comma-separated

    def update_from_worker(self, worker_config: dict):
        """Update settings dynamically from the central Worker module."""
        # Mapping global variable names from the database to this Settings object's attributes.
        key_mapping = {
            "QUALIFICATION_THRESHOLD": "QUALIFICATION_THRESHOLD",
            "AUTO_FORWARD_TO_WRITER": "AUTO_FORWARD_TO_WRITER",
            "SENDER_COMPANY_NAME": "SENDER_COMPANY_NAME",
            "SENDER_ELEVATOR_PITCH": "SENDER_ELEVATOR_PITCH",
            "SENDER_VALUE_PROPS": "SENDER_VALUE_PROPS",
            "OFFER_NAME": "OFFER_NAME",
            "OFFER_SOLUTION_SUMMARY": "OFFER_SOLUTION_SUMMARY",
            "OFFER_CTA": "OFFER_CTA",
            "OFFER_PAIN_POINTS": "OFFER_PAIN_POINTS"
        }
        
        for global_key, local_attr in key_mapping.items():
            if global_key in worker_config:
                val = worker_config[global_key]
                if hasattr(self, local_attr) and val is not None:
                    # Convert types safely if it's strings or floats
                    if isinstance(getattr(self, local_attr), float):
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    elif isinstance(getattr(self, local_attr), bool):
                        if isinstance(val, str):
                            val = val.lower() == 'true'
                        else:
                            val = bool(val)
                            
                    setattr(self, local_attr, val)

    model_config = SettingsConfigDict(
        env_file=("detective/.env", ".env"),
        extra="ignore",
    )


settings = Settings()
