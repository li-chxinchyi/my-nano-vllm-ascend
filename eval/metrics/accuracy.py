import re
from typing import List, Dict
import numpy as np


class AccuracyMetrics:
    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def compute_rouge_l(prediction: str, reference: str) -> float:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(reference, prediction)
            return float(scores['rougeL'].fmeasure)
        except ImportError:
            return AccuracyMetrics._compute_rouge_l_simple(prediction, reference)
    
    @staticmethod
    def _compute_rouge_l_simple(prediction: str, reference: str) -> float:
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        lcs_length = AccuracyMetrics._lcs_length(pred_tokens, ref_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        fmeasure = 2 * precision * recall / (precision + recall)
        return fmeasure
    
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    @staticmethod
    def compute_f1(prediction: str, reference: str) -> float:
        pred_normalized = AccuracyMetrics.normalize_text(prediction)
        ref_normalized = AccuracyMetrics.normalize_text(reference)
        
        pred_tokens = set(pred_normalized.split())
        ref_tokens = set(ref_normalized.split())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = pred_tokens & ref_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    @staticmethod
    def compute_accuracy(prediction: str, reference: str) -> float:
        pred_choice = AccuracyMetrics._extract_choice(prediction)
        ref_choice = AccuracyMetrics._extract_choice(reference)
        
        return 1.0 if pred_choice == ref_choice else 0.0
    
    @staticmethod
    def _extract_choice(text: str) -> str:
        text = text.strip()
        
        pattern = r'\b([A-D])\b'
        match = re.search(pattern, text.upper())
        if match:
            return match.group(1)
        
        choice_keywords = {
            'A': ['a', 'option a', 'choice a', '答案a', '选项a'],
            'B': ['b', 'option b', 'choice b', '答案b', '选项b'],
            'C': ['c', 'option c', 'choice c', '答案c', '选项c'],
            'D': ['d', 'option d', 'choice d', '答案d', '选项d'],
        }
        
        text_lower = text.lower()
        for choice, keywords in choice_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return choice
        
        return 'A'


class AccuracyTracker:
    def __init__(self, metric_type: str = 'f1'):
        self.metric_type = metric_type
        self.scores: List[float] = []
        
        if metric_type not in ['f1', 'rouge_l', 'accuracy']:
            raise ValueError(f"Unsupported metric type: {metric_type}")
    
    def add_sample(self, prediction: str, reference: str):
        if self.metric_type == 'f1':
            score = AccuracyMetrics.compute_f1(prediction, reference)
        elif self.metric_type == 'rouge_l':
            score = AccuracyMetrics.compute_rouge_l(prediction, reference)
        elif self.metric_type == 'accuracy':
            score = AccuracyMetrics.compute_accuracy(prediction, reference)
        else:
            score = 0.0
        
        self.scores.append(score)
    
    def get_summary(self) -> Dict:
        if not self.scores:
            return {
                'metric_type': self.metric_type,
                'num_samples': 0,
                'avg_score': 0.0,
                'median_score': 0.0,
                'p90_score': 0.0,
                'p99_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
            }
        
        return {
            'metric_type': self.metric_type,
            'num_samples': len(self.scores),
            'avg_score': float(np.mean(self.scores)),
            'median_score': float(np.median(self.scores)),
            'p90_score': float(np.percentile(self.scores, 90)) if len(self.scores) > 10 else float(np.max(self.scores)),
            'p99_score': float(np.percentile(self.scores, 99)) if len(self.scores) > 100 else float(np.max(self.scores)),
            'min_score': float(np.min(self.scores)),
            'max_score': float(np.max(self.scores)),
        }
    
    def get_all_scores(self) -> List[float]:
        return self.scores