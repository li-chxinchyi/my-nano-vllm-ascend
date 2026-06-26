import argparse
import subprocess
import sys
import os
import time
from typing import Dict, List

from nanovllm import LLM, SamplingParams
from nanovllm.utils.logger import init_logger

from eval.utils.data_loader import LongBenchDataLoader, get_all_supported_tasks
from eval.metrics.performance import PerformanceTracker
from eval.metrics.accuracy import AccuracyTracker
from eval.utils.report_generator import ReportGenerator

logger = init_logger(__name__)


def check_and_install_dependencies():
    required_packages = [
        ('datasets', 'datasets'),
        ('rouge_score', 'rouge-score'),
        ('nltk', 'nltk'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
    ]
    
    missing_packages = []
    for import_name, pip_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        logger.info(f"Installing missing packages: {missing_packages}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
                logger.info(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {e}")
                logger.error(f"Please manually install: pip install {package}")
                sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="LongBench evaluation script for nano-vLLM-Ascend")
    
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--tasks', type=str, default='longbook_qa_eng,longbook_summ_eng',
                       help='Comma-separated list of tasks to evaluate')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of samples per task')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor parallel size')
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature')
    parser.add_argument('--max_model_len', type=int, default=4096,
                       help='Maximum model length')
    parser.add_argument('--enforce_eager', action='store_true',
                       help='Use eager mode instead of graph mode')
    parser.add_argument('--hccl_port', type=int, default=3456,
                       help='HCCL port for distributed inference')
    parser.add_argument('--skip_dependencies', action='store_true',
                       help='Skip automatic dependency installation')
    parser.add_argument('--max_num_seqs', type=int, default=4,
                       help='Maximum number of sequences per batch')
    
    return parser.parse_args()


def get_memory_usage_mb():
    try:
        import torch
        import torch_npu
        if torch.npu.is_available():
            return torch.npu.memory_allocated() / 1024 / 1024
        elif torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0


def evaluate_task(llm: LLM, task_name: str, samples: List[Dict], 
                  data_loader: LongBenchDataLoader, max_tokens: int, 
                  temperature: float) -> Dict:
    task_config = data_loader.get_task_config(task_name)
    metric_type = task_config.get('metric', 'f1')
    
    perf_tracker = PerformanceTracker()
    acc_tracker = AccuracyTracker(metric_type)
    
    latencies = []
    
    logger.info(f"Evaluating task: {task_name} ({len(samples)} samples)")
    
    for idx, sample in enumerate(samples):
        prompt = data_loader.format_prompt(sample, task_name)
        reference = data_loader.get_reference(sample)
        
        request_id = perf_tracker.start_request()
        
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        
        try:
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            
            perf_tracker.record_first_token(request_id)
            
            output = outputs[0]
            prediction = output['text']
            input_tokens = output['prompt_len']
            output_tokens = len(output['token_ids'])
            
            peak_memory_mb = get_memory_usage_mb()
            
            perf_tracker.end_request(request_id, input_tokens, output_tokens, peak_memory_mb)
            
            acc_tracker.add_sample(prediction, reference)
            
            latencies.append(perf_tracker.metrics[-1].latency_s)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx + 1}/{len(samples)} samples")
        
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            perf_tracker.end_request(request_id, 0, 0, 0.0)
            acc_tracker.scores.append(0.0)
    
    perf_summary = perf_tracker.get_summary()
    acc_summary = acc_tracker.get_summary()
    
    total_time_s = sum(latencies)
    
    return {
        'task_name': task_name,
        'num_samples': len(samples),
        'total_time_s': total_time_s,
        'performance': perf_summary,
        'accuracy': acc_summary,
        'latencies': latencies,
    }


def main():
    args = parse_args()
    
    if not args.skip_dependencies:
        check_and_install_dependencies()
    
    task_list = [t.strip() for t in args.tasks.split(',')]
    
    valid_tasks = get_all_supported_tasks()
    for task in task_list:
        if task not in valid_tasks:
            logger.error(f"Invalid task: {task}. Supported tasks: {valid_tasks}")
            sys.exit(1)
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Tasks: {task_list}")
    logger.info(f"Max samples per task: {args.max_samples}")
    logger.info(f"Output directory: {args.output_dir}")
    
    data_loader = LongBenchDataLoader()
    
    report_generator = ReportGenerator(args.output_dir)
    
    config = {
        'model': args.model,
        'tasks': task_list,
        'max_samples': args.max_samples,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'tensor_parallel_size': args.tensor_parallel_size,
        'max_model_len': args.max_model_len,
        'enforce_eager': args.enforce_eager,
        'hccl_port': args.hccl_port,
    }
    report_generator.save_config(config)
    
    logger.info("Initializing LLM engine...")
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        hccl_port=args.hccl_port,
        max_num_seqs=args.max_num_seqs,
    )
    
    all_results = {}
    
    for task_name in task_list:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting task: {task_name}")
        logger.info(f"{'='*60}")
        
        samples = data_loader.load_dataset(task_name, args.max_samples)
        
        start_time = time.time()
        task_results = evaluate_task(
            llm, task_name, samples, data_loader,
            args.max_tokens, args.temperature
        )
        task_results['evaluation_time_s'] = time.time() - start_time
        
        all_results[task_name] = task_results
        
        logger.info(f"\nTask {task_name} completed:")
        logger.info(f"  - Performance:")
        perf = task_results['performance']
        logger.info(f"    Avg TTFT: {perf['avg_ttft_ms']:.2f}ms")
        logger.info(f"    Avg TPOT: {perf['avg_tpot_ms']:.2f}ms/token")
        logger.info(f"    Avg Throughput: {perf['avg_throughput_tokens_per_s']:.2f} tok/s")
        logger.info(f"  - Accuracy ({task_results['accuracy']['metric_type']}):")
        acc = task_results['accuracy']
        logger.info(f"    Avg Score: {acc['avg_score']:.4f}")
        logger.info(f"    Median Score: {acc['median_score']:.4f}")
    
    report_generator.save_results(all_results)
    
    summary = report_generator.generate_summary(all_results)
    report_generator.save_summary(summary)
    
    report_generator.generate_charts(all_results)
    
    report_generator.generate_markdown_report(config, summary, all_results)
    
    logger.info(f"\n{'='*60}")
    logger.info("Evaluation completed!")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {report_generator.result_dir}")
    logger.info(f"Overall summary:")
    logger.info(f"  - Total samples: {summary['total_samples']}")
    logger.info(f"  - Total time: {summary['total_time_s']:.2f}s")
    logger.info(f"  - Avg TTFT: {summary['overall']['avg_ttft_ms']:.2f}ms")
    logger.info(f"  - Avg Throughput: {summary['overall']['avg_throughput_tokens_per_s']:.2f} tok/s")


if __name__ == "__main__":
    main()