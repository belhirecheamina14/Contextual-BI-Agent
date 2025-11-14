"""
Safe Query Execution Engine - Replaces exec() vulnerability
"""
import ast
import pandas as pd
from typing import Any, Dict, Optional
from enum import Enum

class AllowedOperation(Enum):
    """Whitelist of safe pandas operations"""
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    MAX = "max"
    MIN = "min"
    GROUPBY = "groupby"
    FILTER = "filter"
    SELECT = "select"

class SafeQueryExecutor:
    """
    Validates and executes pandas operations without exec()
    Uses AST parsing and whitelisted operations
    """
    
    ALLOWED_ATTRIBUTES = {
        'sum', 'mean', 'count', 'max', 'min', 'std', 'var',
        'head', 'tail', 'shape', 'columns', 'index'
    }
    
    ALLOWED_METHODS = {
        'groupby', 'agg', 'apply', 'sort_values', 'reset_index'
    }
    
    def __init__(self, df: pd.DataFrame, max_result_size: int = 10000):
        self.df = df
        self.max_result_size = max_result_size
    
    def validate_query(self, query_str: str) -> bool:
        """
        Validates query using AST parsing
        Returns True if safe, raises ValueError if dangerous
        """
        try:
            tree = ast.parse(query_str, mode='eval')
        except SyntaxError:
            raise ValueError("Invalid Python syntax in query")
        
        # Check for dangerous operations
        for node in ast.walk(tree):
            # Block imports
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                raise ValueError("Import statements not allowed")
            
            # Block function calls except whitelisted pandas methods
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    if method_name not in self.ALLOWED_ATTRIBUTES and \
                       method_name not in self.ALLOWED_METHODS:
                        raise ValueError(f"Method '{method_name}' not allowed")
            
            # Block dangerous builtins
            if isinstance(node, ast.Name):
                if node.id in ['exec', 'eval', 'compile', '__import__', 
                              'open', 'input', 'globals', 'locals']:
                    raise ValueError(f"Built-in '{node.id}' not allowed")
        
        return True
    
    def execute(self, query_str: str) -> Dict[str, Any]:
        """
        Safely executes validated query
        Returns structured result with metadata
        """
        # Validate first
        self.validate_query(query_str)
        
        # Create restricted namespace
        namespace = {
            'df': self.df,
            'pd': pd,
            '__builtins__': {}  # Empty builtins
        }
        
        try:
            # Use eval (safer than exec) with restricted namespace
            result = eval(query_str, namespace, {})
            
            # Limit result size
            if isinstance(result, pd.DataFrame) and len(result) > self.max_result_size:
                result = result.head(self.max_result_size)
                truncated = True
            else:
                truncated = False
            
            return {
                "success": True,
                "result": self._format_result(result),
                "truncated": truncated,
                "result_type": type(result).__name__
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def _format_result(self, result: Any) -> Any:
        """Convert result to JSON-serializable format"""
        if isinstance(result, pd.DataFrame):
            return result.to_dict('records')
        elif isinstance(result, pd.Series):
            return result.to_dict()
        elif isinstance(result, (int, float, str, bool)):
            return result
        else:
            return str(result)


# Example usage with LLM-generated code
if __name__ == "__main__":
    # Sample data
    df = pd.DataFrame({
        'Sales': [1000, 2000, 3000],
        'Region': ['North', 'South', 'East'],
        'Product': ['Laptop', 'Monitor', 'Keyboard']
    })
    
    executor = SafeQueryExecutor(df)
    
    # Safe queries
    safe_queries = [
        "df['Sales'].sum()",
        "df[df['Region'] == 'North']['Sales'].mean()",
        "df.groupby('Product')['Sales'].sum()",
    ]
    
    # Dangerous queries (will be rejected)
    dangerous_queries = [
        "__import__('os').system('rm -rf /')",
        "open('/etc/passwd').read()",
        "eval('malicious_code')"
    ]
    
    print("Testing safe queries:")
    for query in safe_queries:
        result = executor.execute(query)
        print(f"Query: {query}")
        print(f"Result: {result}\n")
    
    print("\nTesting dangerous queries (should fail):")
    for query in dangerous_queries:
        try:
            result = executor.execute(query)
            print(f"Query: {query}")
            print(f"Result: {result}\n")
        except ValueError as e:
            print(f"Query: {query}")
            print(f"Blocked: {e}\n")
