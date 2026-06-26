import os
from typing import Dict, List, Optional
from datasets import load_dataset


SUPPORTED_TASKS = {
    'longbook_qa_eng': {
        'metric': 'f1',
        'language': 'en',
        'type': 'qa',
        'description': 'English long document question answering'
    },
    'longbook_qa_chn': {
        'metric': 'f1',
        'language': 'zh',
        'type': 'qa',
        'description': 'Chinese long document question answering'
    },
    'longbook_summ_eng': {
        'metric': 'rouge_l',
        'language': 'en',
        'type': 'summary',
        'description': 'English long document summarization'
    },
    'longbook_summ_chn': {
        'metric': 'rouge_l',
        'language': 'zh',
        'type': 'summary',
        'description': 'Chinese long document summarization'
    },
    'longbook_choice_eng': {
        'metric': 'accuracy',
        'language': 'en',
        'type': 'choice',
        'description': 'English multiple choice questions'
    },
    'longbook_choice_chn': {
        'metric': 'accuracy',
        'language': 'zh',
        'type': 'choice',
        'description': 'Chinese multiple choice questions'
    },
}


class LongBenchDataLoader:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/longbench")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_dataset(self, task_name: str, max_samples: Optional[int] = None) -> List[Dict]:
        if task_name not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task_name}. Supported tasks: {list(SUPPORTED_TASKS.keys())}")
        
        print(f"Loading LongBench dataset: {task_name}")
        try:
            dataset = load_dataset(
                'THUDM/LongBench',
                task_name,
                split='test',
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load dataset from HuggingFace: {e}")
            print("Please manually download the dataset or check your internet connection")
            raise
        
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            print(f"Loaded {max_samples} samples (total available: {len(dataset)})")
        else:
            print(f"Loaded {len(dataset)} samples")
        
        return dataset
    
    def format_prompt(self, sample: Dict, task_name: str) -> str:
        task_config = SUPPORTED_TASKS[task_name]
        task_type = task_config['type']
        language = task_config['language']
        
        if task_type == 'qa':
            return self._format_qa_prompt(sample, language)
        elif task_type == 'summary':
            return self._format_summary_prompt(sample, language)
        elif task_type == 'choice':
            return self._format_choice_prompt(sample, language)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _format_qa_prompt(self, sample: Dict, language: str) -> str:
        context = sample.get('context', '')
        question = sample.get('input', '').split('\n')[-1] if '\n' in sample.get('input', '') else sample.get('input', '')
        
        if language == 'zh':
            prompt = f"请根据以下内容回答问题：\n\n{context}\n\n问题：{question}\n\n答案："
        else:
            prompt = f"Please answer the question based on the following context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        return prompt
    
    def _format_summary_prompt(self, sample: Dict, language: str) -> str:
        context = sample.get('context', '')
        
        if language == 'zh':
            prompt = f"请对以下内容进行总结：\n\n{context}\n\n总结："
        else:
            prompt = f"Please summarize the following content:\n\n{context}\n\nSummary:"
        
        return prompt
    
    def _format_choice_prompt(self, sample: Dict, language: str) -> str:
        context = sample.get('context', '')
        question = sample.get('input', '')
        
        if language == 'zh':
            prompt = f"请根据以下内容选择正确答案：\n\n{context}\n\n{question}\n\n答案："
        else:
            prompt = f"Please choose the correct answer based on the following context:\n\n{context}\n\n{question}\n\nAnswer:"
        
        return prompt
    
    def get_reference(self, sample: Dict) -> str:
        answers = sample.get('answers', [])
        if answers:
            return answers[0] if isinstance(answers, list) else str(answers)
        return ""
    
    def get_task_config(self, task_name: str) -> Dict:
        return SUPPORTED_TASKS.get(task_name, {})


def get_all_supported_tasks() -> List[str]:
    return list(SUPPORTED_TASKS.keys())