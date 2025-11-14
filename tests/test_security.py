import pytest
from backend.app.main import SafeQueryAnalyzer
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

    def test_allowed_queries(self):
        """Test legitimate queries pass validation"""
        safe_queries = [
            "SELECT * FROM sales WHERE region = 'North'",
            "SELECT SUM(sales) FROM sales GROUP BY product",
            "SELECT date, sales FROM sales ORDER BY date DESC LIMIT 100",
        ]

        for query in safe_queries:
            assert SafeQueryAnalyzer.validate_sql(query) is True
