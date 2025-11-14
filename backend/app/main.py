"""
InsightFlow - Production-Ready BI Agent Backend
Complete implementation with security, monitoring, and scalability
"""

# ============================================================================
# PART 1: DEPENDENCIES & CONFIGURATION
# ============================================================================

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from enum import Enum
import asyncpg
from redis import asyncio as aioredis
import jwt
from loguru import logger
import structlog
from prometheus_client import Counter, Histogram, Gauge
from circuitbreaker import circuit
import tenacity
from anthropic import AsyncAnthropic
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings:
    """Centralized configuration"""
    # Database
    DATABASE_URL = "postgresql://user:pass@localhost:5432/insightflow"
    REDIS_URL = "redis://localhost:6379"

    # Security
    JWT_SECRET = "your-secret-key-change-in-production"
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24

    # AI Services
    ANTHROPIC_API_KEY = "your-key-here"

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = 20
    RATE_LIMIT_PER_HOUR = 100

    # Query Execution
    MAX_QUERY_COMPLEXITY = 1000  # AST nodes
    MAX_RESULT_ROWS = 10000
    QUERY_TIMEOUT_SECONDS = 30

    # Caching
    CACHE_TTL_SECONDS = 3600

    # Monitoring
    ENABLE_METRICS = True
    LOG_LEVEL = "INFO"

settings = Settings()

# ============================================================================
# LOGGING & MONITORING SETUP
# ============================================================================

# Structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()

# Prometheus metrics
query_counter = Counter('queries_total', 'Total queries processed', ['status'])
query_duration = Histogram('query_duration_seconds', 'Query processing time')
active_queries = Gauge('active_queries', 'Currently processing queries')
llm_calls = Counter('llm_calls_total', 'LLM API calls', ['provider', 'status'])
cache_hits = Counter('cache_hits_total', 'Cache hit/miss', ['result'])

# ============================================================================
# MODELS & SCHEMAS
# ============================================================================

class QueryComplexity(str, Enum):
    SIMPLE = "simple"      # Single metric, no aggregation
    MODERATE = "moderate"  # Aggregation or filtering
    COMPLEX = "complex"    # Multiple joins or complex logic

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    context_tags: Optional[List[str]] = Field(default=[], max_items=10)
    include_visualization: bool = True
    explain_reasoning: bool = True

    @validator('question')
    def sanitize_question(cls, v):
        # Basic sanitization
        dangerous_patterns = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', '--', '/*']
        for pattern in dangerous_patterns:
            if pattern in v.upper():
                raise ValueError(f"Question contains forbidden pattern: {pattern}")
        return v.strip()

class QueryResponse(BaseModel):
    query_id: str
    answer: str
    data_result: Dict[str, Any]
    visualization: Optional[Dict[str, Any]] = None
    reasoning_chain: Optional[List[str]] = None
    confidence_score: float
    execution_time_ms: float
    sources_used: List[str]
    suggested_followups: List[str]

class User(BaseModel):
    user_id: str
    email: str
    organization: str
    tier: Literal["free", "pro", "enterprise"]
    api_quota_remaining: int

# ============================================================================
# SECURITY & AUTHENTICATION
# ============================================================================

security = HTTPBearer()

class AuthService:
    """JWT-based authentication"""

    @staticmethod
    def create_token(user_id: str, email: str) -> str:
        payload = {
            "user_id": user_id,
            "email": email,
            "exp": datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Dependency for protected routes"""
    payload = AuthService.verify_token(credentials.credentials)

    # Fetch user from database (simplified)
    user = User(
        user_id=payload["user_id"],
        email=payload["email"],
        organization="demo_org",
        tier="pro",
        api_quota_remaining=1000
    )

    return user

# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Redis-based rate limiting"""

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    async def check_rate_limit(self, user_id: str, limit: int, window_seconds: int) -> bool:
        key = f"rate_limit:{user_id}:{window_seconds}"

        current = await self.redis.get(key)

        if current is None:
            await self.redis.setex(key, window_seconds, 1)
            return True

        if int(current) >= limit:
            return False

        await self.redis.incr(key)
        return True

# ============================================================================
# SAFE QUERY EXECUTION ENGINE
# ============================================================================

class SafeQueryAnalyzer:
    """AST-based query validation and complexity analysis"""

    ALLOWED_OPERATIONS = {
        'sum', 'mean', 'count', 'max', 'min', 'std', 'median',
        'groupby', 'agg', 'sort_values', 'head', 'tail',
        'merge', 'join', 'filter', 'select'
    }

    FORBIDDEN_PATTERNS = [
        'exec', 'eval', '__import__', 'compile', 'open',
        'subprocess', 'os.system', 'pickle', 'shelve'
    ]

    @staticmethod
    def analyze_complexity(query_plan: Dict) -> QueryComplexity:
        """Determine query complexity"""
        operations_count = len(query_plan.get('operations', []))
        has_joins = 'join' in str(query_plan).lower()
        has_aggregation = any(op in str(query_plan) for op in ['groupby', 'agg'])

        if operations_count <= 2 and not has_joins:
            return QueryComplexity.SIMPLE
        elif operations_count <= 5 or has_joins:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX

    @staticmethod
    def validate_sql(sql: str) -> bool:
        """Validate SQL query for safety"""
        sql_upper = sql.upper()

        # Block DML/DDL operations
        forbidden_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'TRUNCATE', 'CREATE']

        for keyword in forbidden_keywords:
            if keyword in sql_upper:
                raise ValueError(f"Forbidden SQL keyword: {keyword}")

        # Require SELECT
        if not sql_upper.strip().startswith('SELECT'):
            raise ValueError("Only SELECT queries allowed")

        return True

class QueryExecutor:
    """Safe query execution with timeout and resource limits"""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type(asyncpg.exceptions.ConnectionDoesNotExistError)
    )
    async def execute_sql(self, sql: str, params: Dict = None) -> pd.DataFrame:
        """Execute SQL with safety checks"""
        # Validate first
        SafeQueryAnalyzer.validate_sql(sql)

        async with self.db_pool.acquire() as conn:
            # Set query timeout
            await conn.execute(f"SET statement_timeout = {settings.QUERY_TIMEOUT_SECONDS * 1000}")

            try:
                rows = await conn.fetch(sql, *(params.values() if params else []))

                # Convert to DataFrame
                if rows:
                    df = pd.DataFrame([dict(row) for row in rows])

                    # Limit result size
                    if len(df) > settings.MAX_RESULT_ROWS:
                        log.warning("Result truncated", rows=len(df), limit=settings.MAX_RESULT_ROWS)
                        df = df.head(settings.MAX_RESULT_ROWS)

                    return df
                else:
                    return pd.DataFrame()

            except asyncpg.exceptions.QueryCanceledError:
                raise HTTPException(status_code=408, detail="Query timeout exceeded")
            except Exception as e:
                log.error("Query execution failed", error=str(e), sql=sql)
                raise HTTPException(status_code=400, detail=f"Query execution error: {str(e)}")

# ============================================================================
# AI ORCHESTRATION WITH CLAUDE
# ============================================================================

class AIOrchestrator:
    """Intelligent query orchestration with Claude"""

    def __init__(self, anthropic_client: AsyncAnthropic):
        self.client = anthropic_client
        self.model = "claude-sonnet-4-20250514"

    @circuit(failure_threshold=3, recovery_timeout=60)
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        stop=tenacity.stop_after_attempt(3)
    )
    async def generate_sql(self, question: str, schema: Dict) -> Dict[str, Any]:
        """Convert natural language to SQL"""

        prompt = f"""You are a SQL expert. Convert this question to a safe SELECT query.

Database Schema:
{json.dumps(schema, indent=2)}

Question: {question}

Return ONLY valid SQL. No explanations. Use parameterized queries where appropriate.
Only return SELECT statements. No DML/DDL operations."""

        llm_calls.labels(provider='anthropic', status='attempt').inc()

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            sql = response.content[0].text.strip()

            # Clean up markdown if present
            if sql.startswith("```sql"):
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif sql.startswith("```"):
                sql = sql.split("```")[1].split("```")[0].strip()

            llm_calls.labels(provider='anthropic', status='success').inc()

            return {
                "sql": sql,
                "confidence": 0.85,  # Could use actual model confidence
                "reasoning": "SQL generated from natural language"
            }

        except Exception as e:
            llm_calls.labels(provider='anthropic', status='error').inc()
            log.error("SQL generation failed", error=str(e))
            raise

    async def generate_explanation(self, question: str, data: pd.DataFrame, context: str) -> Dict[str, Any]:
        """Generate human-friendly explanation of results"""

        # Sample data for context
        data_sample = data.head(5).to_json(orient='records') if not data.empty else "[]"
        data_summary = {
            "rows": len(data),
            "columns": list(data.columns) if not data.empty else [],
            "sample": data_sample
        }

        prompt = f"""You are a business intelligence analyst. Explain these results clearly.

Question: {question}

Data Summary:
{json.dumps(data_summary, indent=2)}

Business Context:
{context}

Provide:
1. A clear, concise answer
2. Key insights (2-3 bullet points)
3. Any caveats or limitations
4. 2 suggested follow-up questions

Format as JSON with keys: answer, insights, caveats, followups"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text.strip()

            # Parse JSON response
            if result_text.startswith("```json"):
                result_text = result_text.split("```json")[1].split("```")[0].strip()

            return json.loads(result_text)

        except Exception as e:
            log.error("Explanation generation failed", error=str(e))
            return {
                "answer": "Results retrieved successfully",
                "insights": ["Data analysis completed"],
                "caveats": ["Automated explanation unavailable"],
                "followups": ["What other metrics would you like to see?"]
            }

# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

class VisualizationEngine:
    """Automatic chart generation based on data characteristics"""

    @staticmethod
    def auto_visualize(df: pd.DataFrame, question: str) -> Optional[Dict]:
        """Automatically choose and generate appropriate visualization"""

        if df.empty or len(df) == 0:
            return None

        # Determine chart type based on data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        fig = None

        # Time series detection
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]

        if date_cols and numeric_cols:
            # Time series line chart
            fig = px.line(df, x=date_cols[0], y=numeric_cols[0],
                         title=f"{numeric_cols[0]} over Time")

        elif len(categorical_cols) == 1 and len(numeric_cols) == 1:
            # Bar chart
            fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0],
                        title=f"{numeric_cols[0]} by {categorical_cols[0]}")

        elif len(numeric_cols) >= 2:
            # Scatter plot
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                           title=f"{numeric_cols[1]} vs {numeric_cols[0]}")

        elif len(categorical_cols) == 1:
            # Pie chart for distributions
            value_counts = df[categorical_cols[0]].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index,
                        title=f"Distribution of {categorical_cols[0]}")

        if fig:
            return {
                "type": "plotly",
                "data": json.loads(fig.to_json())
            }

        return None

# ============================================================================
# CACHING LAYER
# ============================================================================

class CacheService:
    """Redis-based intelligent caching"""

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    @staticmethod
    def generate_cache_key(question: str, schema_version: str) -> str:
        """Generate deterministic cache key"""
        content = f"{question}:{schema_version}"
        return f"query_cache:{hashlib.sha256(content.encode()).hexdigest()}"

    async def get(self, key: str) -> Optional[Dict]:
        """Get cached result"""
        try:
            cached = await self.redis.get(key)
            if cached:
                cache_hits.labels(result='hit').inc()
                return json.loads(cached)
            cache_hits.labels(result='miss').inc()
            return None
        except Exception as e:
            log.error("Cache retrieval failed", error=str(e))
            return None

    async def set(self, key: str, value: Dict, ttl: int = settings.CACHE_TTL_SECONDS):
        """Cache result with TTL"""
        try:
            await self.redis.setex(key, ttl, json.dumps(value))
        except Exception as e:
            log.error("Cache storage failed", error=str(e))

# ============================================================================
# ANOMALY DETECTION
# ============================================================================

class AnomalyDetector:
    """Proactive anomaly detection in query results"""

    @staticmethod
    def detect_anomalies(df: pd.DataFrame) -> List[Dict]:
        """Detect statistical anomalies in data"""
        anomalies = []

        for col in df.select_dtypes(include=[np.number]).columns:
            try:
                values = df[col].dropna().values.reshape(-1, 1)

                if len(values) < 10:
                    continue

                # Use Isolation Forest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                predictions = iso_forest.fit_predict(values)

                anomaly_indices = np.where(predictions == -1)[0]

                if len(anomaly_indices) > 0:
                    anomalies.append({
                        "column": col,
                        "anomaly_count": len(anomaly_indices),
                        "anomaly_values": df.iloc[anomaly_indices][col].tolist()[:5],  # Top 5
                        "severity": "high" if len(anomaly_indices) > len(df) * 0.2 else "medium"
                    })

            except Exception as e:
                log.warning("Anomaly detection failed", column=col, error=str(e))

        return anomalies

# ============================================================================
# MAIN APPLICATION
# ============================================================================

app = FastAPI(
    title="InsightFlow API",
    description="Production-ready BI Agent with AI-powered insights",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients (initialized on startup)
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[aioredis.Redis] = None
anthropic_client: Optional[AsyncAnthropic] = None

# Service instances
rate_limiter: Optional[RateLimiter] = None
cache_service: Optional[CacheService] = None
query_executor: Optional[QueryExecutor] = None
ai_orchestrator: Optional[AIOrchestrator] = None

# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    global db_pool, redis_client, anthropic_client
    global rate_limiter, cache_service, query_executor, ai_orchestrator

    log.info("Starting InsightFlow services...")

    # Database pool
    db_pool = await asyncpg.create_pool(settings.DATABASE_URL, min_size=5, max_size=20)

    # Redis
    redis_client = await aioredis.from_url(settings.REDIS_URL)

    # Anthropic
    anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    # Initialize services
    rate_limiter = RateLimiter(redis_client)
    cache_service = CacheService(redis_client)
    query_executor = QueryExecutor(db_pool)
    ai_orchestrator = AIOrchestrator(anthropic_client)

    log.info("All services started successfully")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    log.info("Shutting down services...")

    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()

    log.info("Shutdown complete")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

@app.post("/api/v2/query", response_model=QueryResponse)
@query_duration.time()
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Main query endpoint - processes natural language BI questions
    """
    active_queries.inc()
    start_time = datetime.utcnow()
    query_id = hashlib.sha256(f"{current_user.user_id}:{request.question}:{start_time}".encode()).hexdigest()[:16]

    try:
        # Rate limiting
        allowed = await rate_limiter.check_rate_limit(
            current_user.user_id,
            settings.RATE_LIMIT_PER_MINUTE,
            60
        )

        if not allowed:
            query_counter.labels(status='rate_limited').inc()
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        log.info("Processing query", query_id=query_id, user=current_user.user_id, question=request.question)

        # Check cache
        cache_key = cache_service.generate_cache_key(request.question, "v1")
        cached_result = await cache_service.get(cache_key)

        if cached_result:
            log.info("Cache hit", query_id=query_id)
            query_counter.labels(status='cache_hit').inc()
            active_queries.dec()
            return QueryResponse(**cached_result)

        # Get database schema (simplified - would be more sophisticated in production)
        schema = {
            "tables": {
                "sales": ["date", "region", "product", "sales", "cost", "units", "customer_segment"]
            }
        }

        # Generate SQL using AI
        sql_result = await ai_orchestrator.generate_sql(request.question, schema)
        sql_query = sql_result["sql"]

        # Execute query
        data = await query_executor.execute_sql(sql_query)

        # Parallel processing: explanation + visualization + anomaly detection
        tasks = [
            ai_orchestrator.generate_explanation(request.question, data, "Q1 2024 Sales Data"),
        ]

        if request.include_visualization:
            tasks.append(asyncio.to_thread(VisualizationEngine.auto_visualize, data, request.question))

        if len(data) > 0:
            tasks.append(asyncio.to_thread(AnomalyDetector.detect_anomalies, data))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        explanation = results[0] if not isinstance(results[0], Exception) else {}
        visualization = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
        anomalies = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else []

        # Build response
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        response = QueryResponse(
            query_id=query_id,
            answer=explanation.get("answer", "Query executed successfully"),
            data_result={
                "rows": len(data),
                "columns": list(data.columns),
                "data": data.head(100).to_dict('records'),  # Limit preview
                "anomalies": anomalies
            },
            visualization=visualization,
            reasoning_chain=[
                f"Translated question to SQL: {sql_query[:100]}...",
                f"Executed query, returned {len(data)} rows",
                "Generated explanation and insights",
                "Performed anomaly detection"
            ] if request.explain_reasoning else None,
            confidence_score=sql_result.get("confidence", 0.8),
            execution_time_ms=execution_time,
            sources_used=["sales_database", "business_context"],
            suggested_followups=explanation.get("followups", [])
        )

        # Cache result
        background_tasks.add_task(cache_service.set, cache_key, response.dict())

        # Log successful query
        background_tasks.add_task(
            log_query_metrics,
            query_id, current_user.user_id, request.question, "success", execution_time
        )

        query_counter.labels(status='success').inc()
        active_queries.dec()

        return response

    except HTTPException:
        active_queries.dec()
        raise
    except Exception as e:
        active_queries.dec()
        query_counter.labels(status='error').inc()
        log.error("Query processing failed", query_id=query_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

async def log_query_metrics(query_id: str, user_id: str, question: str, status: str, duration_ms: float):
    """Background task to log query metrics"""
    try:
        # Would insert into analytics database
        log.info("Query completed",
                query_id=query_id,
                user_id=user_id,
                status=status,
                duration_ms=duration_ms)
    except Exception as e:
        log.error("Failed to log metrics", error=str(e))

# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@app.post("/api/v2/admin/clear-cache")
async def clear_cache(current_user: User = Depends(get_current_user)):
    """Clear query cache (admin only)"""
    if current_user.tier != "enterprise":
        raise HTTPException(status_code=403, detail="Admin access required")

    # Clear cache
    await redis_client.flushdb()
    return {"status": "cache_cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
