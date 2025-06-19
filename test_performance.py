#!/usr/bin/env python3
"""
Performance testing script for the optimized AI Symptom Checker API
"""

import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict
import argparse

class PerformanceTester:
    """Test performance of the optimized API"""
    
    def __init__(self, base_url: str = "http://localhost:5000", api_key: str = "test-api-key"):
        self.base_url = base_url
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health(self) -> Dict:
        """Test health endpoint"""
        start_time = time.time()
        async with self.session.get(f"{self.base_url}/health") as response:
            duration = time.time() - start_time
            return {
                "status": response.status,
                "duration": duration,
                "data": await response.json() if response.status == 200 else None
            }
    
    async def test_symptom_analysis(self, symptom: str) -> Dict:
        """Test symptom analysis endpoint"""
        start_time = time.time()
        data = {"symptom": symptom}
        
        async with self.session.post(
            f"{self.base_url}/api/analyze-symptom",
            headers=self.headers,
            json=data
        ) as response:
            duration = time.time() - start_time
            return {
                "status": response.status,
                "duration": duration,
                "data": await response.json() if response.status == 200 else None
            }
    
    async def test_batch_analysis(self, symptoms: List[str]) -> Dict:
        """Test batch analysis endpoint"""
        start_time = time.time()
        
        async with self.session.post(
            f"{self.base_url}/api/batch-analyze",
            headers=self.headers,
            json=symptoms
        ) as response:
            duration = time.time() - start_time
            return {
                "status": response.status,
                "duration": duration,
                "data": await response.json() if response.status == 200 else None
            }
    
    async def test_concurrent_requests(self, symptoms: List[str], concurrency: int = 10) -> Dict:
        """Test concurrent requests"""
        start_time = time.time()
        
        # Create tasks for concurrent execution
        tasks = []
        for symptom in symptoms[:concurrency]:
            task = self.test_symptom_analysis(symptom)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_duration = time.time() - start_time
        
        # Process results
        successful_results = [r for r in results if isinstance(r, dict) and r["status"] == 200]
        failed_results = [r for r in results if not isinstance(r, dict) or r["status"] != 200]
        
        durations = [r["duration"] for r in successful_results]
        
        return {
            "total_duration": total_duration,
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "avg_duration": statistics.mean(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "throughput": len(successful_results) / total_duration if total_duration > 0 else 0
        }
    
    async def test_cache_performance(self, symptom: str, iterations: int = 5) -> Dict:
        """Test cache performance by repeating the same request"""
        durations = []
        
        for i in range(iterations):
            result = await self.test_symptom_analysis(symptom)
            if result["status"] == 200:
                durations.append(result["duration"])
        
        return {
            "iterations": iterations,
            "avg_duration": statistics.mean(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "cache_effectiveness": "Cache working" if durations and durations[0] > durations[-1] else "No cache effect"
        }

async def run_performance_tests():
    """Run comprehensive performance tests"""
    test_symptoms = [
        "I have a severe headache",
        "My chest hurts and I feel short of breath",
        "I have a fever of 102 degrees",
        "I feel very tired and weak",
        "I have nausea and can't keep food down",
        "I have a persistent dry cough",
        "I have a mild headache",
        "My stomach feels upset",
        "I have back pain",
        "I feel dizzy and lightheaded"
    ]
    
    print("🚀 Starting Performance Tests")
    print("=" * 50)
    
    async with PerformanceTester() as tester:
        # Test 1: Health Check
        print("\n1. Health Check Test")
        health_result = await tester.test_health()
        print(f"   Status: {health_result['status']}")
        print(f"   Duration: {health_result['duration']:.3f}s")
        
        if health_result['status'] != 200:
            print("   ❌ Health check failed - API not available")
            return
        
        print("   ✅ Health check passed")
        
        # Test 2: Single Symptom Analysis
        print("\n2. Single Symptom Analysis Test")
        single_result = await tester.test_symptom_analysis(test_symptoms[0])
        print(f"   Status: {single_result['status']}")
        print(f"   Duration: {single_result['duration']:.3f}s")
        
        if single_result['status'] == 200:
            data = single_result['data']
            print(f"   Category: {data.get('symptom_category', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 0):.2f}")
            print(f"   Severity: {data.get('severity', 'N/A')}")
            print("   ✅ Single analysis successful")
        else:
            print("   ❌ Single analysis failed")
        
        # Test 3: Batch Analysis
        print("\n3. Batch Analysis Test")
        batch_result = await tester.test_batch_analysis(test_symptoms[:5])
        print(f"   Status: {batch_result['status']}")
        print(f"   Duration: {batch_result['duration']:.3f}s")
        
        if batch_result['status'] == 200:
            results = batch_result['data'].get('results', [])
            print(f"   Processed {len(results)} symptoms")
            print("   ✅ Batch analysis successful")
        else:
            print("   ❌ Batch analysis failed")
        
        # Test 4: Concurrent Requests
        print("\n4. Concurrent Requests Test")
        concurrency = 5
        concurrent_result = await tester.test_concurrent_requests(test_symptoms, concurrency)
        print(f"   Total Duration: {concurrent_result['total_duration']:.3f}s")
        print(f"   Successful: {concurrent_result['successful_requests']}/{concurrency}")
        print(f"   Failed: {concurrent_result['failed_requests']}")
        print(f"   Average Duration: {concurrent_result['avg_duration']:.3f}s")
        print(f"   Throughput: {concurrent_result['throughput']:.2f} requests/second")
        
        if concurrent_result['successful_requests'] == concurrency:
            print("   ✅ Concurrent requests successful")
        else:
            print("   ⚠️  Some concurrent requests failed")
        
        # Test 5: Cache Performance
        print("\n5. Cache Performance Test")
        cache_result = await tester.test_cache_performance(test_symptoms[0], 5)
        print(f"   Iterations: {cache_result['iterations']}")
        print(f"   Average Duration: {cache_result['avg_duration']:.3f}s")
        print(f"   Min Duration: {cache_result['min_duration']:.3f}s")
        print(f"   Max Duration: {cache_result['max_duration']:.3f}s")
        print(f"   Cache Effectiveness: {cache_result['cache_effectiveness']}")
        
        # Performance Summary
        print("\n" + "=" * 50)
        print("📊 Performance Summary")
        print("=" * 50)
        
        if single_result['status'] == 200:
            print(f"Single Request Latency: {single_result['duration']:.3f}s")
        
        if batch_result['status'] == 200:
            batch_latency = batch_result['duration'] / len(test_symptoms[:5])
            print(f"Batch Request Latency: {batch_latency:.3f}s per request")
        
        if concurrent_result['successful_requests'] > 0:
            print(f"Concurrent Throughput: {concurrent_result['throughput']:.2f} requests/second")
        
        # Performance Assessment
        print("\n🎯 Performance Assessment")
        if single_result['duration'] < 1.0:
            print("✅ Excellent: Single request latency < 1s")
        elif single_result['duration'] < 2.0:
            print("✅ Good: Single request latency < 2s")
        else:
            print("⚠️  Needs improvement: Single request latency > 2s")
        
        if concurrent_result['throughput'] > 10:
            print("✅ Excellent: Throughput > 10 requests/second")
        elif concurrent_result['throughput'] > 5:
            print("✅ Good: Throughput > 5 requests/second")
        else:
            print("⚠️  Needs improvement: Throughput < 5 requests/second")

def main():
    parser = argparse.ArgumentParser(description='Performance testing for Symptom Checker API')
    parser.add_argument('--url', default='http://localhost:5000', help='API base URL')
    parser.add_argument('--api-key', default='test-api-key', help='API key for authentication')
    
    args = parser.parse_args()
    
    print(f"Testing API at: {args.url}")
    print(f"Using API key: {args.api_key}")
    
    asyncio.run(run_performance_tests())

if __name__ == "__main__":
    main() 