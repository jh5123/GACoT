"""
Dependency Extraction
Extract calculation dependencies from text
"""

import re
from typing import Dict, List, Set


class DependencyExtractor:
    """Extract calculation dependencies from text."""
    
    def __init__(self):
        """Initialize extractor with patterns."""
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> List:
        """Compile regex patterns for efficiency."""
        return [
            # Variable = expression
            re.compile(r'(\w+)\s*=\s*([^=\n]+?)(?:\s*=\s*[\$\d]|$)', re.IGNORECASE),
            # Variable: expression  
            re.compile(r'(\w+)\s*:\s*([^:\n]+?)(?:\s*=\s*[\$\d]|$)', re.IGNORECASE),
            # Calculate Variable as expression
            re.compile(r'(?:calculate|compute)\s+(\w+)\s+(?:as|=)\s+([^=\n]+)', re.IGNORECASE),
        ]
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract dependencies from text.
        
        Args:
            text: Text containing calculations
            
        Returns:
            Dictionary mapping variables to their dependencies
        """
        # Normalize text
        text = self._normalize_text(text)
        
        # Try multiple extraction strategies
        deps1 = self._extract_with_patterns(text)
        deps2 = self._extract_natural_language(text)
        deps3 = self._extract_calculation_blocks(text)
        
        # Merge all found dependencies
        merged = {}
        for deps in [deps1, deps2, deps3]:
            for var, var_deps in deps.items():
                if var not in merged:
                    merged[var] = []
                merged[var].extend(var_deps)
                merged[var] = list(set(merged[var]))  # Remove duplicates
        
        return merged
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for extraction."""
        # Replace Unicode characters
        replacements = {
            '×': '*',
            '÷': '/',
            '−': '-',
            '–': '-',
            '—': '-',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _extract_with_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract using compiled patterns."""
        dependencies = {}
        
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                target = self._normalize_var(match.group(1))
                formula = match.group(2)
                
                # Extract variables from formula
                deps = self._extract_variables(formula)
                
                if deps and target:
                    dependencies[target] = deps
        
        return dependencies
    
    def _extract_natural_language(self, text: str) -> Dict[str, List[str]]:
        """Extract from natural language descriptions."""
        dependencies = {}
        
        patterns = [
            r'(\w+)\s+(?:depends on|is based on|calculated from|derived from)\s+([^\.]+)',
            r'(\w+)\s+(?:uses|requires|needs)\s+([^\.]+)',
            r'To (?:calculate|compute|find)\s+(\w+)[,\s]+(?:we need|use|require)\s+([^\.]+)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                target = self._normalize_var(match.group(1))
                deps_text = match.group(2)
                
                # Extract variables from dependency text
                deps = self._extract_variables(deps_text)
                
                if deps and target:
                    dependencies[target] = deps
        
        return dependencies
    
    def _extract_calculation_blocks(self, text: str) -> Dict[str, List[str]]:
        """Extract from calculation blocks."""
        dependencies = {}
        
        lines = text.split('\n')
        for line in lines:
            # Skip comments and empty lines
            if line.strip().startswith('#') or not line.strip():
                continue
            
            # Look for calculations
            if '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    # Extract target variable
                    target_match = re.search(r'\b([A-Z][a-zA-Z_]*|[a-z]+_[a-z_]+)\b', left)
                    if target_match:
                        target = self._normalize_var(target_match.group(1))
                        
                        # Extract source variables
                        deps = self._extract_variables(right)
                        
                        if deps:
                            dependencies[target] = deps
        
        return dependencies
    
    def _normalize_var(self, var: str) -> str:
        """Normalize variable name."""
        if not var:
            return ""
        
        # Clean up variable name
        var = var.strip()
        
        # Convert spaces to underscores
        var = var.replace(' ', '_')
        
        # Remove special characters except underscores
        var = re.sub(r'[^a-zA-Z0-9_]', '', var)
        
        return var
    
    def _extract_variables(self, formula: str) -> List[str]:
        """Extract variable names from formula."""
        if not formula:
            return []
        
        variables = []
        
        # Remove numbers, currency symbols, and operators
        clean_formula = re.sub(r'[\$£€¥]\d+[MBK]?|\d+\.?\d*[MBK]?|[%]', '', formula)
        clean_formula = re.sub(r'[+\-*/()%,]', ' ', clean_formula)
        
        # Find variable patterns
        # Matches: Revenue, EBITDA, Tax_Rate, total_sources, etc.
        patterns = [
            r'\b[A-Z][a-zA-Z_]*\b',  # CamelCase or UPPERCASE
            r'\b[a-z]+(?:_[a-z]+)+\b',  # snake_case
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, clean_formula):
                var = self._normalize_var(match.group())
                if var and len(var) > 1:  # Skip single letters
                    variables.append(var)
        
        return list(set(variables))  # Remove duplicates