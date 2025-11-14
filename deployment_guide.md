# InsightFlow: Complete Deployment Guide

## 🚀 Quick Start (Production-Ready)

### Prerequisites
- Docker & Docker Compose installed
- Anthropic API key
- 8GB RAM minimum
- 20GB disk space

### 1. Initial Setup (5 minutes)

```bash
# Clone repository
git clone https://github.com/your-org/insightflow.git
cd insightflow

# Create environment file
cat > .env << EOF
# Database
DB_PASSWORD=$(openssl rand -hex 32)

# AI Services
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Security
JWT_SECRET=$(openssl rand -hex 32)

# Admin Access
GRAFANA_PASSWORD=admin_password_here

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
EOF

# Create required directories
mkdir -p logs database/backups monitoring/grafana/{dashboards,datasources}
```

### 2. Database Initialization

Create `database/init.sql`:

```sql
-- InsightFlow Database Schema

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    organization VARCHAR(255),
    tier VARCHAR(50) DEFAULT 'free',
    api_quota_daily INT DEFAULT 100,
    api_quota_remaining INT DEFAULT 100,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Sales data table (example business data)
CREATE TABLE IF NOT EXISTS sales_transactions (
    transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL,
    region VARCHAR(100),
    product VARCHAR(100),
    sales DECIMAL(12,2),
    cost DECIMAL(12,2),
    units INTEGER,
    customer_segment VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_date_region (date, region),
    INDEX idx_product (product),
    INDEX idx_segment (customer_segment)
);

-- Query logs for analytics
CREATE TABLE IF NOT EXISTS query_logs (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    question TEXT NOT NULL,
    sql_generated TEXT,
    execution_time_ms FLOAT,
    status VARCHAR(50),
    error_message TEXT,
    result_rows INT,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_user_date (user_id, created_at),
    INDEX idx_status (status)
);

-- Context knowledge base
CREATE TABLE IF NOT EXISTS knowledge_base (
    kb_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100),
    tags TEXT[],
    embedding VECTOR(1536),  -- For vector similarity search
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User feedback for continuous improvement
CREATE TABLE IF NOT EXISTS query_feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES query_logs(query_id),
    user_id UUID REFERENCES users(user_id),
    rating INT CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert sample data
INSERT INTO sales_transactions (date, region, product, sales, cost, units, customer_segment)
SELECT
    date '2024-01-01' + (i % 365) * interval '1 day' as date,
    CASE (i % 4)
        WHEN 0 THEN 'North'
        WHEN 1 THEN 'South'
        WHEN 2 THEN 'East'
        ELSE 'West'
    END as region,
    CASE (i % 3)
        WHEN 0 THEN 'Laptop'
        WHEN 1 THEN 'Monitor'
        ELSE 'Keyboard'
    END as product,
    10000 + (random() * 5000)::decimal(12,2) as sales,
    5000 + (random() * 2500)::decimal(12,2) as cost,
    (50 + random() * 50)::int as units,
    CASE (i % 3)
        WHEN 0 THEN 'Enterprise'
        WHEN 1 THEN 'SMB'
        ELSE 'Consumer'
    END as customer_segment
FROM generate_series(1, 10000) as i;

-- Create indexes for performance
CREATE INDEX idx_sales_date ON sales_transactions(date);
CREATE INDEX idx_sales_region_product ON sales_transactions(region, product);

-- Create materialized view for common aggregations (performance optimization)
CREATE MATERIALIZED VIEW sales_summary AS
SELECT
    date_trunc('day', date) as date,
    region,
    product,
    customer_segment,
    SUM(sales) as total_sales,
    SUM(cost) as total_cost,
    SUM(units) as total_units,
    COUNT(*) as transaction_count
FROM sales_transactions
GROUP BY date_trunc('day', date), region, product, customer_segment;

-- Create refresh function
CREATE OR REPLACE FUNCTION refresh_sales_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW sales_summary;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO insightflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO insightflow;
```

### 3. Monitoring Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'insightflow-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

Create `monitoring/grafana/datasources/prometheus.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
```

### 4. Deploy Stack

```bash
# Build and start all services
docker-compose -f docker-compose-production.yml up -d --build

# Wait for services to be healthy
docker-compose -f docker-compose-production.yml ps

# Check logs
docker-compose -f docker-compose-production.yml logs -f api

# Verify health
curl http://localhost:8000/health
```

### 5. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend | http://localhost | N/A |
| API | http://localhost:8000 | Bearer token |
| API Docs | http://localhost:8000/docs | N/A |
| Grafana | http://localhost:3000 | admin / [GRAFANA_PASSWORD] |
| Prometheus | http://localhost:9090 | N/A |

---

## 🧪 Comprehensive Testing Suite

### Unit Tests

Create `tests/test_security.py`:

```python
import pytest
from backend.app.core.data_query import SafeQueryAnalyzer, QueryExecutor
import asyncpg

class TestSecurity:
    """Security-focused test suite"""

    def test_sql_injection_prevention(self):
        """Test SQL injection attempts are blocked"""
        dangerous_queries = [
            "SELECT * FROM users; DROP TABLE users;--",
            "SELECT * FROM sales WHERE 1=1 OR 1=1",
            "'; DELETE FROM sales; --",
            "UNION SELECT password FROM users",
        ]

        for query in dangerous_queries:
            with pytest.raises(ValueError):
                SafeQueryAnalyzer.validate_sql(query)

    def test_code_injection_prevention(self):
        """Test code injection via exec/eval is blocked"""
        dangerous_code = [
            "__import__('os').system('rm -rf /')",
            "eval('malicious_code')",
            "exec('import os; os.system(\"evil\")')",
        ]

        analyzer = SafeQueryAnalyzer()
        for code in dangerous_code:
            with pytest.raises(ValueError):
                analyzer.validate_query(code)

    def test_allowed_queries(self):
        """Test legitimate queries pass validation"""
        safe_queries = [
            "SELECT * FROM sales WHERE region = 'North'",
            "SELECT SUM(sales) FROM sales GROUP BY product",
            "SELECT date, sales FROM sales ORDER BY date DESC LIMIT 100",
        ]

        for query in safe_queries:
            assert SafeQueryAnalyzer.validate_sql(query) is True

@pytest.mark.asyncio
class TestQueryExecutor:
    """Test query execution with timeouts and limits"""

    async def test_query_timeout(self):
        """Test long-running queries are cancelled"""
        # Mock slow query
        slow_query = "SELECT pg_sleep(60); SELECT * FROM sales;"

        with pytest.raises(HTTPException) as exc:
            await executor.execute_sql(slow_query)

        assert exc.value.status_code == 408

    async def test_result_size_limit(self):
        """Test results are truncated at limit"""
        query = "SELECT * FROM sales"
        result = await executor.execute_sql(query)

        assert len(result) <= 10000  # MAX_RESULT_ROWS

class TestRateLimiting:
    """Test rate limiting functionality"""

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self):
        """Test rate limits block excessive requests"""
        user_id = "test-user"

        # Should allow first 20 requests
        for i in range(20):
            allowed = await rate_limiter.check_rate_limit(user_id, 20, 60)
            assert allowed is True

        # 21st request should be blocked
        allowed = await rate_limiter.check_rate_limit(user_id, 20, 60)
        assert allowed is False
```

### Integration Tests

Create `tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

class TestAPIEndpoints:
    """Integration tests for API endpoints"""

    def test_health_check(self):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_query_without_auth(self):
        """Test query endpoint requires authentication"""
        response = client.post("/api/v2/query", json={
            "question": "What is the total revenue?"
        })
        assert response.status_code == 401

    def test_query_with_auth(self):
        """Test successful query with authentication"""
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        response = client.post("/api/v2/query",
            json={"question": "What is the total revenue?"},
            headers=headers
        )
        assert response.status_code == 200
        assert "answer" in response.json()
        assert "data_result" in response.json()

    def test_invalid_question(self):
        """Test validation of invalid questions"""
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}

        # Too short
        response = client.post("/api/v2/query",
            json={"question": "hi"},
            headers=headers
        )
        assert response.status_code == 422

        # Contains SQL keywords
        response = client.post("/api/v2/query",
            json={"question": "DROP TABLE users"},
            headers=headers
        )
        assert response.status_code == 422

class TestCaching:
    """Test caching functionality"""

    def test_cache_hit(self):
        """Test cached responses are returned"""
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        question = {"question": "What is the total revenue for 2024?"}

        # First request
        response1 = client.post("/api/v2/query", json=question, headers=headers)
        time1 = response1.json()["execution_time_ms"]

        # Second request (should hit cache)
        response2 = client.post("/api/v2/query", json=question, headers=headers)
        time2 = response2.json()["execution_time_ms"]

        # Cached response should be faster
        assert time2 < time1 * 0.5
```

### Load Testing

Create `tests/load_test.py`:

```python
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

# Run with: locust -f tests/load_test.py --host=http://localhost:8000
# Target: 100 users, 10 req/sec sustained
```

### Performance Benchmarks

Create `tests/benchmark.py`:

```python
import asyncio
import time
from statistics import mean, stdev

async def benchmark_query_performance():
    """Benchmark query execution performance"""

    test_queries = [
        "What is the total revenue?",
        "Show sales by region",
        "Which product has highest profit margin?",
        "Analyze sales trends over time",
    ]

    results = {}

    for query in test_queries:
        times = []
        for _ in range(10):  # 10 iterations
            start = time.time()
            # Execute query
            await execute_test_query(query)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        results[query] = {
            "mean": mean(times),
            "stdev": stdev(times),
            "min": min(times),
            "max": max(times),
            "p95": sorted(times)[int(len(times) * 0.95)]
        }

    # Assert performance targets
    for query, metrics in results.items():
        assert metrics["p95"] < 5000, f"Query too slow: {query} (p95: {metrics['p95']}ms)"
        print(f"✓ {query}: {metrics['mean']:.0f}ms avg, {metrics['p95']:.0f}ms p95")

if __name__ == "__main__":
    asyncio.run(benchmark_query_performance())
```

### Run All Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov locust

# Run unit tests with coverage
pytest tests/ --cov=backend --cov-report=html --cov-report=term

# Run security tests only
pytest tests/test_security.py -v

# Run load test (separate terminal)
locust -f tests/load_test.py --host=http://localhost:8000 --users 100 --spawn-rate 10

# Run benchmarks
python tests/benchmark.py
```

---

## 📊 Performance Targets

| Metric | Target | Acceptable | Failure |
|--------|--------|-----------|---------|
| p50 Latency | < 1s | < 2s | > 3s |
| p95 Latency | < 3s | < 5s | > 8s |
| p99 Latency | < 5s | < 10s | > 15s |
| Throughput | > 100 req/s | > 50 req/s | < 20 req/s |
| Error Rate | < 0.1% | < 1% | > 2% |
| Cache Hit Rate | > 40% | > 20% | < 10% |

---

## 🔐 Security Checklist

- [x] SQL injection prevention (AST validation)
- [x] Code injection prevention (no exec/eval)
- [x] JWT authentication required
- [x] Rate limiting enforced
- [x] Query timeout protection
- [x] Result size limits
- [x] Input sanitization
- [x] Secrets in environment variables
- [x] HTTPS enforced (in production)
- [x] CORS properly configured
- [ ] WAF deployed (CloudFlare/AWS WAF)
- [ ] DDoS protection
- [ ] Audit logging enabled
- [ ] Penetration testing completed
- [ ] SOC 2 compliance (for enterprise)

---

## 🚢 Production Deployment (Kubernetes)

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: insightflow-api
  labels:
    app: insightflow
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: insightflow
  template:
    metadata:
      labels:
        app: insightflow
    spec:
      containers:
      - name: api
        image: insightflow/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: insightflow-secrets
              key: database-url
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: insightflow-secrets
              key: anthropic-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: insightflow-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: insightflow
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: insightflow-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: insightflow-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

Deploy to Kubernetes:

```bash
# Create secrets
kubectl create secret generic insightflow-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=anthropic-key="sk-ant-..."

# Deploy
kubectl apply -f k8s/

# Check status
kubectl get pods -l app=insightflow
kubectl get svc insightflow-api

# Scale manually if needed
kubectl scale deployment insightflow-api --replicas=5
```

---

## 📈 Monitoring & Alerting

### Key Metrics to Monitor

1. **Application Metrics**
   - Query success rate
   - Average response time
   - LLM API latency
   - Cache hit rate
   - Active queries

2. **Infrastructure Metrics**
   - CPU/Memory usage
   - Database connections
   - Redis memory usage
   - Disk I/O

3. **Business Metrics**
   - Daily active users
   - Queries per user
   - API cost per query
   - User satisfaction (ratings)

### Alert Rules

Create `monitoring/alerts.yml`:

```yaml
groups:
  - name: insightflow_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(queries_total{status="error"}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"

      - alert: SlowQueries
        expr: histogram_quantile(0.95, query_duration_seconds) > 5
        for: 5m
        annotations:
          summary: "p95 query latency exceeds 5s"

      - alert: LowCacheHitRate
        expr: rate(cache_hits_total{result="hit"}[10m]) / rate(cache_hits_total[10m]) < 0.2
        for: 10m
        annotations:
          summary: "Cache hit rate below 20%"
```

---

## 🎯 Success Criteria

### MVP Ready Checklist
- [x] Core query functionality works
- [x] Security vulnerabilities fixed
- [x] Authentication implemented
- [x] Rate limiting active
- [x] Monitoring deployed
- [x] Load tested (100 users)
- [ ] User documentation complete
- [ ] Admin dashboard functional

### Production Ready Checklist
- [ ] All MVP items complete
- [ ] 95%+ test coverage
- [ ] Security audit passed
- [ ] Load tested (1000+ users)
- [ ] Disaster recovery tested
- [ ] SLA defined and monitored
- [ ] Customer support trained
- [ ] Legal/compliance approved

**Estimated Timeline**: 6-8 weeks to Production-Ready
