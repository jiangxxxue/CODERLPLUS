import ast
import subprocess
import tempfile
import os
from typing import Dict, Any


class SimpleCodeFilter:
    """Simple filter for code quality and execution"""
    
    def __init__(self, max_cyclomatic_complexity: int = 15):
        self.max_cyclomatic_complexity = max_cyclomatic_complexity
    
    def calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity of Python code"""
        try:
            tree = ast.parse(code)
            complexity = 1
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    complexity += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
        except:
            return 999
    
    def check_compilation(self, code: str) -> bool:
        """Check if code compiles without syntax errors"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def check_execution(self, code: str, test_input: str = "1\n") -> bool:
        """Check if code can execute without errors"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            try:
                result = subprocess.run(
                    ['python', temp_path],
                    input=test_input,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=3
                )
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                return False
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        except:
            return False
    
    def filter_code(self, code: str, test_input: str = "1\n") -> Dict[str, Any]:
        """Apply all filters to the code"""
        
        if not code or not code.strip():
            return {
                'passed_all': False,
                'reason': 'Empty code',
                'compiles': False,
                'executes': False,
                'complexity': 999
            }
        
        compiles = self.check_compilation(code)
        if not compiles:
            return {
                'passed_all': False,
                'reason': 'Syntax error',
                'compiles': False,
                'executes': False,
                'complexity': 999
            }
        
        executes = self.check_execution(code, test_input)
        if not executes:
            return {
                'passed_all': False,
                'reason': 'Execution error',
                'compiles': True,
                'executes': False,
                'complexity': 999
            }
        
        complexity = self.calculate_cyclomatic_complexity(code)
        if complexity > self.max_cyclomatic_complexity:
            return {
                'passed_all': False,
                'reason': f'Complexity too high: {complexity} > {self.max_cyclomatic_complexity}',
                'compiles': True,
                'executes': True,
                'complexity': complexity
            }
        
        return {
            'passed_all': True,
            'reason': 'All checks passed',
            'compiles': True,
            'executes': True,
            'complexity': complexity
        }