import logging
from prometheus_client import Counter, start_http_server

logger = logging.getLogger(__name__)

class Telemetry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Telemetry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # Requests Counter
        self.llm_requests_total = Counter(
            'llm_requests_total',
            'Total number of LLM requests',
            ['model', 'mode', 'status']
        )
        
        # Tokens Counter
        self.llm_tokens_total = Counter(
            'llm_tokens_total',
            'Total number of tokens used',
            ['model', 'mode', 'token_type']
        )
        
        self._initialized = True

    def record_request(self, model: str, mode: str, status: str = "success"):
        self.llm_requests_total.labels(model=model, mode=mode, status=status).inc()

    def record_tokens(self, model: str, mode: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        if prompt_tokens > 0:
            self.llm_tokens_total.labels(model=model, mode=mode, token_type='prompt').inc(prompt_tokens)
        if completion_tokens > 0:
            self.llm_tokens_total.labels(model=model, mode=mode, token_type='completion').inc(completion_tokens)

    def start_server(self, port: int = 8000):
        try:
            start_http_server(port)
            logger.info(f"📊 Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"❌ Failed to start metrics server: {e}")

# Global instance
telemetry = Telemetry()
