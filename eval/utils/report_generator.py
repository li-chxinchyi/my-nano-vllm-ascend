import os
import json
from datetime import datetime
from typing import Dict, List
import numpy as np


class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = os.path.join(output_dir, self.timestamp)
        os.makedirs(self.result_dir, exist_ok=True)
    
    def save_config(self, config: Dict):
        config_path = os.path.join(self.result_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Config saved to {config_path}")
    
    def save_results(self, results: Dict):
        results_path = os.path.join(self.result_dir, "results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {results_path}")
    
    def generate_summary(self, results: Dict) -> Dict:
        summary = {
            'timestamp': self.timestamp,
            'output_dir': self.result_dir,
            'total_samples': sum(r.get('num_samples', 0) for r in results.values()),
            'total_time_s': sum(r.get('total_time_s', 0) for r in results.values()),
            'tasks': {}
        }
        
        for task_name, task_results in results.items():
            perf = task_results.get('performance', {})
            acc = task_results.get('accuracy', {})
            
            summary['tasks'][task_name] = {
                'num_samples': task_results.get('num_samples', 0),
                'performance': {
                    'avg_ttft_ms': perf.get('avg_ttft_ms', 0),
                    'avg_tpot_ms': perf.get('avg_tpot_ms', 0),
                    'avg_latency_s': perf.get('avg_latency_s', 0),
                    'avg_throughput_tokens_per_s': perf.get('avg_throughput_tokens_per_s', 0),
                    'peak_memory_mb': perf.get('peak_memory_mb', 0),
                },
                'accuracy': {
                    'metric_type': acc.get('metric_type', 'unknown'),
                    'avg_score': acc.get('avg_score', 0),
                }
            }
        
        all_ttfts = [perf.get('avg_ttft_ms', 0) for r in results.values() for perf in [r.get('performance', {})]]
        all_tpots = [perf.get('avg_tpot_ms', 0) for r in results.values() for perf in [r.get('performance', {})] if perf.get('avg_tpot_ms', 0) > 0]
        all_throughputs = [perf.get('avg_throughput_tokens_per_s', 0) for r in results.values() for perf in [r.get('performance', {})]]
        
        summary['overall'] = {
            'avg_ttft_ms': float(np.mean(all_ttfts)) if all_ttfts else 0,
            'avg_tpot_ms': float(np.mean(all_tpots)) if all_tpots else 0,
            'avg_throughput_tokens_per_s': float(np.mean(all_throughputs)) if all_throughputs else 0,
        }
        
        return summary
    
    def save_summary(self, summary: Dict):
        summary_path = os.path.join(self.result_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to {summary_path}")
    
    def generate_charts(self, results: Dict):
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            task_names = list(results.keys())
            if not task_names:
                return
            
            self._generate_performance_chart(results, task_names)
            self._generate_accuracy_chart(results, task_names)
            self._generate_latency_distribution(results, task_names)
            
        except ImportError:
            print("matplotlib not available, skipping chart generation")
    
    def _generate_performance_chart(self, results: Dict, task_names: List[str]):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Performance Metrics Comparison', fontsize=16)
        
        ttfts = [results[t]['performance'].get('avg_ttft_ms', 0) for t in task_names]
        tpots = [results[t]['performance'].get('avg_tpot_ms', 0) for t in task_names]
        throughputs = [results[t]['performance'].get('avg_throughput_tokens_per_s', 0) for t in task_names]
        latencies = [results[t]['performance'].get('avg_latency_s', 0) for t in task_names]
        
        axes[0, 0].bar(task_names, ttfts, color='skyblue')
        axes[0, 0].set_title('Average TTFT (ms)')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(task_names, tpots, color='lightcoral')
        axes[0, 1].set_title('Average TPOT (ms/token)')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 0].bar(task_names, throughputs, color='lightgreen')
        axes[1, 0].set_title('Average Throughput (tokens/s)')
        axes[1, 0].set_ylabel('Tokens/s')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(task_names, latencies, color='gold')
        axes[1, 1].set_title('Average Latency (s)')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        chart_path = os.path.join(self.result_dir, "performance_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Performance chart saved to {chart_path}")
    
    def _generate_accuracy_chart(self, results: Dict, task_names: List[str]):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scores = [results[t]['accuracy'].get('avg_score', 0) for t in task_names]
        metric_types = [results[t]['accuracy'].get('metric_type', 'unknown') for t in task_names]
        
        colors = {'f1': 'blue', 'rouge_l': 'green', 'accuracy': 'red'}
        bar_colors = [colors.get(m, 'gray') for m in metric_types]
        
        bars = ax.bar(task_names, scores, color=bar_colors)
        ax.set_title('Accuracy Scores by Task', fontsize=16)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=c, label=l) 
                          for l, c in colors.items() if l in metric_types]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        chart_path = os.path.join(self.result_dir, "accuracy_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Accuracy chart saved to {chart_path}")
    
    def _generate_latency_distribution(self, results: Dict, task_names: List[str]):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        latency_data = []
        valid_tasks = []
        for task in task_names:
            latencies = results[task].get('latencies', [])
            if latencies:
                latency_data.append(latencies)
                valid_tasks.append(task)
        
        if not latency_data:
            return
        
        ax.boxplot(latency_data, labels=valid_tasks)
        ax.set_title('Latency Distribution by Task', fontsize=16)
        ax.set_ylabel('Latency (s)')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        chart_path = os.path.join(self.result_dir, "latency_distribution.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Latency distribution chart saved to {chart_path}")
    
    def generate_markdown_report(self, config: Dict, summary: Dict, results: Dict):
        report_path = os.path.join(self.result_dir, "report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# LongBench Evaluation Report\n\n")
            
            f.write("## 1. Test Configuration\n\n")
            f.write(f"- **Model**: `{config.get('model', 'N/A')}`\n")
            f.write(f"- **Tasks**: {', '.join(config.get('tasks', []))}\n")
            f.write(f"- **Max Samples**: {config.get('max_samples', 'N/A')} per task\n")
            f.write(f"- **Max Tokens**: {config.get('max_tokens', 'N/A')}\n")
            f.write(f"- **Temperature**: {config.get('temperature', 'N/A')}\n")
            f.write(f"- **Tensor Parallel Size**: {config.get('tensor_parallel_size', 'N/A')}\n")
            f.write(f"- **Timestamp**: {self.timestamp}\n\n")
            
            f.write("## 2. Overall Summary\n\n")
            overall = summary.get('overall', {})
            f.write(f"- **Total Samples**: {summary.get('total_samples', 0)}\n")
            f.write(f"- **Total Time**: {summary.get('total_time_s', 0):.2f}s\n")
            f.write(f"- **Average TTFT**: {overall.get('avg_ttft_ms', 0):.2f}ms\n")
            f.write(f"- **Average TPOT**: {overall.get('avg_tpot_ms', 0):.2f}ms/token\n")
            f.write(f"- **Average Throughput**: {overall.get('avg_throughput_tokens_per_s', 0):.2f} tokens/s\n\n")
            
            f.write("## 3. Performance Metrics by Task\n\n")
            f.write("| Task | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | Latency (s) | Peak Memory (MB) |\n")
            f.write("|------|-----------|-----------|---------------------|-------------|------------------|\n")
            
            for task_name in config.get('tasks', []):
                if task_name in summary.get('tasks', {}):
                    task_perf = summary['tasks'][task_name]['performance']
                    f.write(f"| {task_name} | {task_perf['avg_ttft_ms']:.2f} | {task_perf['avg_tpot_ms']:.2f} | "
                           f"{task_perf['avg_throughput_tokens_per_s']:.2f} | {task_perf['avg_latency_s']:.2f} | "
                           f"{task_perf['peak_memory_mb']:.0f} |\n")
            
            f.write("\n## 4. Accuracy Scores by Task\n\n")
            f.write("| Task | Metric Type | Avg Score | Median | P90 | P99 |\n")
            f.write("|------|-------------|-----------|--------|-----|-----|\n")
            
            for task_name in config.get('tasks', []):
                if task_name in results:
                    acc = results[task_name]['accuracy']
                    f.write(f"| {task_name} | {acc['metric_type']} | {acc['avg_score']:.3f} | "
                           f"{acc['median_score']:.3f} | {acc['p90_score']:.3f} | {acc['p99_score']:.3f} |\n")
            
            f.write("\n## 5. Charts\n\n")
            f.write("### Performance Comparison\n")
            f.write("![Performance Chart](performance_chart.png)\n\n")
            f.write("### Accuracy Comparison\n")
            f.write("![Accuracy Chart](accuracy_chart.png)\n\n")
            f.write("### Latency Distribution\n")
            f.write("![Latency Distribution](latency_distribution.png)\n\n")
            
            f.write("## 6. Detailed Statistics\n\n")
            for task_name in config.get('tasks', []):
                if task_name in results:
                    perf = results[task_name]['performance']
                    f.write(f"### {task_name}\n\n")
                    f.write("**Performance Details**:\n")
                    f.write(f"- Total Input Tokens: {perf['total_input_tokens']}\n")
                    f.write(f"- Total Output Tokens: {perf['total_output_tokens']}\n")
                    f.write(f"- Number of Requests: {perf['num_requests']}\n")
                    f.write(f"- P90 TTFT: {perf['p90_ttft_ms']:.2f}ms\n")
                    f.write(f"- P99 TTFT: {perf['p99_ttft_ms']:.2f}ms\n")
                    f.write(f"- P90 Latency: {perf['p90_latency_s']:.2f}s\n")
                    f.write(f"- P99 Latency: {perf['p99_latency_s']:.2f}s\n\n")
        
        print(f"Markdown report saved to {report_path}")