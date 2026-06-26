import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class PerformanceMetrics:
    request_id: int
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    latency_s: float
    throughput_tokens_per_s: float
    peak_memory_mb: float


class PerformanceTracker:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self._request_start_times: Dict[int, float] = {}
        self._first_token_times: Dict[int, float] = {}
        self._request_counter: int = 0
        
    def start_request(self) -> int:
        request_id = self._request_counter
        self._request_counter += 1
        self._request_start_times[request_id] = time.perf_counter()
        return request_id
    
    def record_first_token(self, request_id: int):
        if request_id not in self._first_token_times:
            self._first_token_times[request_id] = time.perf_counter()
    
    def end_request(self, request_id: int, input_tokens: int, output_tokens: int, peak_memory_mb: float):
        end_time = time.perf_counter()
        
        start_time = self._request_start_times.get(request_id)
        first_token_time = self._first_token_times.get(request_id)
        
        if start_time is None:
            print(f"Warning: No start time for request {request_id}")
            return
        
        latency_s = end_time - start_time
        
        if first_token_time is not None and output_tokens > 0:
            ttft_ms = (first_token_time - start_time) * 1000
            if output_tokens > 1:
                tpot_ms = ((end_time - first_token_time) / (output_tokens - 1)) * 1000
            else:
                tpot_ms = 0.0
        else:
            ttft_ms = latency_s * 1000
            tpot_ms = 0.0
        
        throughput_tokens_per_s = output_tokens / latency_s if latency_s > 0 else 0.0
        
        metrics = PerformanceMetrics(
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            latency_s=latency_s,
            throughput_tokens_per_s=throughput_tokens_per_s,
            peak_memory_mb=peak_memory_mb
        )
        
        self.metrics.append(metrics)
        
        if request_id in self._request_start_times:
            del self._request_start_times[request_id]
        if request_id in self._first_token_times:
            del self._first_token_times[request_id]
    
    def get_summary(self) -> Dict:
        if not self.metrics:
            return {
                'num_requests': 0,
                'avg_ttft_ms': 0.0,
                'median_ttft_ms': 0.0,
                'p90_ttft_ms': 0.0,
                'p99_ttft_ms': 0.0,
                'avg_tpot_ms': 0.0,
                'median_tpot_ms': 0.0,
                'avg_latency_s': 0.0,
                'median_latency_s': 0.0,
                'p90_latency_s': 0.0,
                'p99_latency_s': 0.0,
                'avg_throughput_tokens_per_s': 0.0,
                'median_throughput_tokens_per_s': 0.0,
                'peak_memory_mb': 0.0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
            }
        
        ttfts = [m.ttft_ms for m in self.metrics]
        tpots = [m.tpot_ms for m in self.metrics if m.tpot_ms > 0]
        latencies = [m.latency_s for m in self.metrics]
        throughputs = [m.throughput_tokens_per_s for m in self.metrics]
        peak_memory = [m.peak_memory_mb for m in self.metrics]
        
        total_input_tokens = sum(m.input_tokens for m in self.metrics)
        total_output_tokens = sum(m.output_tokens for m in self.metrics)
        
        return {
            'num_requests': len(self.metrics),
            'avg_ttft_ms': float(np.mean(ttfts)),
            'median_ttft_ms': float(np.median(ttfts)),
            'p90_ttft_ms': float(np.percentile(ttfts, 90)) if len(ttfts) > 10 else float(np.max(ttfts)),
            'p99_ttft_ms': float(np.percentile(ttfts, 99)) if len(ttfts) > 100 else float(np.max(ttfts)),
            'avg_tpot_ms': float(np.mean(tpots)) if tpots else 0.0,
            'median_tpot_ms': float(np.median(tpots)) if tpots else 0.0,
            'avg_latency_s': float(np.mean(latencies)),
            'median_latency_s': float(np.median(latencies)),
            'p90_latency_s': float(np.percentile(latencies, 90)) if len(latencies) > 10 else float(np.max(latencies)),
            'p99_latency_s': float(np.percentile(latencies, 99)) if len(latencies) > 100 else float(np.max(latencies)),
            'avg_throughput_tokens_per_s': float(np.mean(throughputs)),
            'median_throughput_tokens_per_s': float(np.median(throughputs)),
            'peak_memory_mb': float(np.max(peak_memory)),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
        }
    
    def get_all_ttfts(self) -> List[float]:
        return [m.ttft_ms for m in self.metrics]
    
    def get_all_tpots(self) -> List[float]:
        return [m.tpot_ms for m in self.metrics if m.tpot_ms > 0]
    
    def get_all_latencies(self) -> List[float]:
        return [m.latency_s for m in self.metrics]
    
    def get_all_throughputs(self) -> List[float]:
        return [m.throughput_tokens_per_s for m in self.metrics]