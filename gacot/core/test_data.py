"""
Test Data Management
Load and generate test problems for evaluation
"""

import json
from pathlib import Path
from typing import List, Dict, Any


class TestDataManager:
    """Manage test data for evaluations."""
    
    def __init__(self):
        """Initialize test data manager."""
        self.problems = []
    
    def load_problems(self, dataset_path: str) -> List[Dict]:
        """
        Load test problems from dataset.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List of test problems
        """
        dataset_path = Path(dataset_path)
        
        if dataset_path.exists():
            problems = self._load_from_files(dataset_path)
            if problems:
                return problems
        
        print("  Using synthetic test problems")
        return self.generate_synthetic_problems()
    
    def _load_from_files(self, dataset_path: Path) -> List[Dict]:
        """Load problems from JSONL files."""
        problems = []
        
        jsonl_files = list(dataset_path.glob("**/*.jsonl"))
        json_files = list(dataset_path.glob("**/*.json"))
        
        all_files = jsonl_files + json_files
        
        for file_path in all_files:
            try:
                if file_path.suffix == '.jsonl':
                    problems.extend(self._load_jsonl(file_path))
                else:
                    problems.extend(self._load_json(file_path))
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
        
        return problems
    
    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        """Load problems from JSONL file."""
        problems = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    problem = json.loads(line)
                    problem = self._validate_problem(problem)
                    problems.append(problem)
                except json.JSONDecodeError as e:
                    print(f"  Line {line_num} in {file_path}: {e}")
        
        return problems
    
    def _load_json(self, file_path: Path) -> List[Dict]:
        """Load problems from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                return [self._validate_problem(p) for p in data]
            else:
                return [self._validate_problem(data)]
        except Exception:
            return []
    
    def _validate_problem(self, problem: Dict) -> Dict:
        """Validate and fix problem structure."""
        if "question" not in problem:
            problem["question"] = "Calculate the financial model"
        
        if "dependencies" not in problem:
            problem["dependencies"] = self._generate_default_dependencies()
        
        deps = problem["dependencies"]
        if "graph" not in deps:
            deps["graph"] = {}
        if "cascades" not in deps:
            deps["cascades"] = []
        
        if "expected_values" not in problem:
            problem["expected_values"] = {}
        
        return problem
    
    def _generate_default_dependencies(self) -> Dict:
        """Generate default dependency structure."""
        return {
            "graph": {
                "EBITDA": ["Revenue", "Costs"],
                "Net_Income": ["EBITDA", "Tax_Rate"],
                "Valuation": ["Net_Income", "Multiple"]
            },
            "cascades": [
                {
                    "change": "Revenue",
                    "affects": ["EBITDA", "Net_Income", "Valuation"]
                },
                {
                    "change": "Tax_Rate",
                    "affects": ["Net_Income", "Valuation"]
                }
            ]
        }
    
    def generate_synthetic_problems(self) -> List[Dict]:
        """Generate synthetic test problems with ground truth values."""
        problems = []
        
        # Problem 1: Basic P&L Model
        problems.append({
            "question": "Calculate EBITDA, Net Income, and Enterprise Value",
            "initial_values": {
                "Revenue": 100000000,
                "Operating_Costs": 60000000,
                "Depreciation": 5000000,
                "Interest": 3000000,
                "Tax_Rate": 0.25,
                "EV_Multiple": 10
            },
            "dependencies": {
                "graph": {
                    "EBITDA": ["Revenue", "Operating_Costs"],
                    "EBIT": ["EBITDA", "Depreciation"],
                    "EBT": ["EBIT", "Interest"],
                    "Net_Income": ["EBT", "Tax_Rate"],
                    "Enterprise_Value": ["EBITDA", "EV_Multiple"]
                },
                "cascades": [
                    {
                        "change": "Revenue",
                        "affects": ["EBITDA", "EBIT", "EBT", "Net_Income", "Enterprise_Value"]
                    },
                    {
                        "change": "Tax_Rate",
                        "affects": ["Net_Income"]
                    },
                    {
                        "change": "EV_Multiple",
                        "affects": ["Enterprise_Value"]
                    }
                ]
            },
            "expected_values": {
                "EBITDA": 40000000,
                "EBIT": 35000000,
                "EBT": 32000000,
                "Net_Income": 24000000,
                "Enterprise_Value": 400000000
            }
        })
        
        # Problem 2: Simple DCF Model
        problems.append({
            "question": "Calculate Free Cash Flow and present value for Year 1",
            "initial_values": {
                "EBITDA": 50000000,
                "Tax_Rate": 0.30,
                "CapEx": 10000000,
                "NWC_Change": 5000000,
                "WACC": 0.10
            },
            "dependencies": {
                "graph": {
                    "Tax": ["EBITDA", "Tax_Rate"],
                    "FCF": ["EBITDA", "Tax", "CapEx", "NWC_Change"],
                    "PV_FCF": ["FCF", "WACC"]
                },
                "cascades": [
                    {
                        "change": "EBITDA",
                        "affects": ["Tax", "FCF", "PV_FCF"]
                    },
                    {
                        "change": "WACC",
                        "affects": ["PV_FCF"]
                    }
                ]
            },
            "expected_values": {
                "Tax": 15000000,
                "FCF": 20000000,
                "PV_FCF": 18181818
            }
        })
        
        # Problem 3: Leverage Ratios
        problems.append({
            "question": "Calculate Debt, Interest Coverage Ratio, and Debt/EBITDA",
            "initial_values": {
                "EBITDA": 80000000,
                "Interest_Rate": 0.05,
                "Target_Leverage": 3.0,
                "EBIT": 70000000
            },
            "dependencies": {
                "graph": {
                    "Debt": ["EBITDA", "Target_Leverage"],
                    "Interest_Expense": ["Debt", "Interest_Rate"],
                    "Interest_Coverage": ["EBIT", "Interest_Expense"],
                    "Debt_to_EBITDA": ["Debt", "EBITDA"]
                },
                "cascades": [
                    {
                        "change": "Target_Leverage",
                        "affects": ["Debt", "Interest_Expense", "Interest_Coverage", "Debt_to_EBITDA"]
                    },
                    {
                        "change": "Interest_Rate",
                        "affects": ["Interest_Expense", "Interest_Coverage"]
                    }
                ]
            },
            "expected_values": {
                "Debt": 240000000,
                "Interest_Expense": 12000000,
                "Interest_Coverage": 5.833333,
                "Debt_to_EBITDA": 3.0
            }
        })
        
        # Problem 4: Working Capital Model
        problems.append({
            "question": "Calculate Working Capital metrics and Cash Conversion Cycle",
            "initial_values": {
                "Revenue": 365000000,
                "COGS": 255500000,
                "Accounts_Receivable": 50000000,
                "Inventory": 35000000,
                "Accounts_Payable": 30000000
            },
            "dependencies": {
                "graph": {
                    "DSO": ["Accounts_Receivable", "Revenue"],
                    "DIO": ["Inventory", "COGS"],
                    "DPO": ["Accounts_Payable", "COGS"],
                    "Working_Capital": ["Accounts_Receivable", "Inventory", "Accounts_Payable"],
                    "Cash_Conversion_Cycle": ["DSO", "DIO", "DPO"]
                },
                "cascades": [
                    {
                        "change": "Revenue",
                        "affects": ["DSO", "Cash_Conversion_Cycle"]
                    },
                    {
                        "change": "Inventory",
                        "affects": ["DIO", "Working_Capital", "Cash_Conversion_Cycle"]
                    }
                ]
            },
            "expected_values": {
                "DSO": 50.0,
                "DIO": 50.0,
                "DPO": 42.86,
                "Working_Capital": 55000000,
                "Cash_Conversion_Cycle": 57.14
            }
        })
        
        # Problem 5: Return Metrics
        problems.append({
            "question": "Calculate ROE, ROA, and ROIC",
            "initial_values": {
                "Net_Income": 30000000,
                "Total_Assets": 500000000,
                "Total_Equity": 200000000,
                "Debt": 300000000,
                "EBIT": 45000000,
                "Tax_Rate": 0.30
            },
            "dependencies": {
                "graph": {
                    "ROE": ["Net_Income", "Total_Equity"],
                    "ROA": ["Net_Income", "Total_Assets"],
                    "NOPAT": ["EBIT", "Tax_Rate"],
                    "Invested_Capital": ["Total_Equity", "Debt"],
                    "ROIC": ["NOPAT", "Invested_Capital"]
                },
                "cascades": [
                    {
                        "change": "Net_Income",
                        "affects": ["ROE", "ROA"]
                    },
                    {
                        "change": "EBIT",
                        "affects": ["NOPAT", "ROIC"]
                    }
                ]
            },
            "expected_values": {
                "ROE": 0.15,
                "ROA": 0.06,
                "NOPAT": 31500000,
                "Invested_Capital": 500000000,
                "ROIC": 0.063
            }
        })
        
        # Problem 6: LBO with Circular Reference
        problems.append({
            "question": "Calculate LBO sources and uses with debt financing fees (note: circular reference)",
            "initial_values": {
                "Purchase_Price": 1000000000,
                "Equity_Investment": 400000000,
                "Fee_Percentage": 0.02
            },
            "dependencies": {
                "graph": {
                    "Debt": ["Purchase_Price", "Equity_Investment", "Fees"],
                    "Fees": ["Debt", "Fee_Percentage"],
                    "Total_Sources": ["Equity_Investment", "Debt"],
                    "Total_Uses": ["Purchase_Price", "Fees"]
                },
                "cascades": [
                    {
                        "change": "Fee_Percentage",
                        "affects": ["Fees", "Debt", "Total_Sources", "Total_Uses"]
                    }
                ],
                "has_circular": True
            },
            "expected_values": {
                "Debt": 612244898,
                "Fees": 12244898,
                "Total_Sources": 1012244898,
                "Total_Uses": 1012244898
            }
        })
        
        return problems