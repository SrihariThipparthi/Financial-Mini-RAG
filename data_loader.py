import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json

class DataLoader:
    def __init__(self, faqs_path: str, performance_path: str):
        self.faqs_path = faqs_path
        self.performance_path = performance_path
        
    def load_faqs(self) -> List[Dict[str, Any]]:
        try:
            df = pd.read_csv(self.faqs_path)
            documents = []
            
            for idx, row in df.iterrows():
                doc = {
                    'id': f"faq_{idx}",
                    'content': f"Question: {row['question']}\nAnswer: {row['answer']}",
                    'type': 'faq',
                    'metadata': {
                        'question': row['question'],
                        'answer': row['answer']
                    }
                }
                documents.append(doc)
                
            print(f"Loaded {len(documents)} FAQs")
            return documents
        except Exception as e:
            print(f"Error loading FAQs: {e}")
            return []
    
    def load_performance_data(self) -> List[Dict[str, Any]]:
        try:
            df = pd.read_csv(self.performance_path)
            documents = []
            
            for _, row in df.iterrows():
                fund_text = f"{row['fund_name']} ({row['category']}) has "
                
                metrics = []
                if 'cagr_3yr (%)' in row:
                    metrics.append(f"3-year CAGR: {row['cagr_3yr (%)']}%")
                if 'volatility (%)' in row:
                    metrics.append(f"volatility: {row['volatility (%)']}%")
                if 'sharpe_ratio' in row:
                    metrics.append(f"Sharpe ratio: {row['sharpe_ratio']}")
                
                fund_text += ", ".join(metrics)
                
                doc = {
                    'id': f"fund_{row['fund_id']}",
                    'content': fund_text,
                    'type': 'fund',
                    'metadata': {
                        'fund_id': row['fund_id'],
                        'fund_name': row['fund_name'],
                        'category': row['category'],
                        'cagr_3yr': row['cagr_3yr (%)'],
                        'volatility': row['volatility (%)'],
                        'sharpe_ratio': row['sharpe_ratio']
                    }
                }
                documents.append(doc)
                
            print(f"Loaded {len(documents)} funds")
            return documents
        except Exception as e:
            print(f"Error loading performance data: {e}")
            return []
    
    def load_all_data(self) -> List[Dict[str, Any]]:
        faqs = self.load_faqs()
        funds = self.load_performance_data()
        all_data = faqs + funds
        print(f"Total documents loaded: {len(all_data)}")
        return all_data