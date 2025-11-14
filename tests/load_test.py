from locust import HttpUser, task, between

class InsightFlowUser(HttpUser):
    """Simulate user behavior for load testing"""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Login and get token"""
        self.token = "test-token"  # In real test, get from auth endpoint
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(3)
    def query_simple(self):
        """Simple query (most common)"""
        self.client.post("/api/v2/query",
            json={"question": "What is total revenue?"},
            headers=self.headers
        )

    @task(2)
    def query_with_viz(self):
        """Query with visualization"""
        self.client.post("/api/v2/query",
            json={
                "question": "Show sales by region",
                "include_visualization": True
            },
            headers=self.headers
        )

    @task(1)
    def query_complex(self):
        """Complex query"""
        self.client.post("/api/v2/query",
            json={
                "question": "Compare revenue by product and region for last quarter",
                "include_visualization": True,
                "explain_reasoning": True
            },
            headers=self.headers
        )

    @task(1)
    def health_check(self):
        """Health check"""
        self.client.get("/health")
