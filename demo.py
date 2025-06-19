#!/usr/bin/env python3
"""
Demo script for AI Symptom Checker
"""

import json
import requests
import time
from typing import Dict, List

class SymptomCheckerDemo:
    """Demo class for testing the symptom checker"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session_id = f"demo_session_{int(time.time())}"
    
    def test_health(self) -> bool:
        """Test health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Health check passed: {data['message']}")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to server. Make sure the server is running with: python app.py")
            return False
    
    def test_symptom_analysis(self, symptom: str) -> Dict:
        """Test symptom analysis endpoint"""
        try:
            data = {
                'symptom': symptom,
                'session_id': self.session_id
            }
            
            response = requests.post(
                f"{self.base_url}/api/analyze-symptom",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"✗ Symptom analysis failed: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"✗ Error testing symptom analysis: {e}")
            return {}
    
    def test_chat(self, message: str, conversation_history: List[Dict] = None) -> Dict:
        """Test chat endpoint"""
        try:
            if conversation_history is None:
                conversation_history = []
            
            data = {
                'message': message,
                'session_id': self.session_id,
                'conversation_history': conversation_history
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"✗ Chat failed: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"✗ Error testing chat: {e}")
            return {}
    
    def run_demo(self):
        """Run the complete demo"""
        print("="*60)
        print("AI Symptom Checker Demo")
        print("="*60)
        
        # Test health
        if not self.test_health():
            return
        
        print("\n" + "="*60)
        print("Testing Symptom Analysis")
        print("="*60)
        
        # Test different symptoms
        test_symptoms = [
            "I have a severe headache that started this morning",
            "My chest hurts and I feel short of breath",
            "I have a fever of 102 degrees",
            "I feel very tired and weak",
            "I have nausea and can't keep food down",
            "I have a persistent dry cough"
        ]
        
        for symptom in test_symptoms:
            print(f"\n🔍 Analyzing: {symptom}")
            result = self.test_symptom_analysis(symptom)
            
            if result:
                print(f"   Category: {result.get('symptom_category', 'Unknown')}")
                print(f"   Severity: {result.get('severity', 'Unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
                print(f"   Questions: {len(result.get('follow_up_questions', []))}")
                print(f"   Recommendations: {len(result.get('recommendations', []))}")
        
        print("\n" + "="*60)
        print("Testing Chat Interface")
        print("="*60)
        
        # Test chat conversation
        conversation_history = []
        
        chat_messages = [
            "I have a headache",
            "It started this morning and it's quite severe",
            "Yes, I also feel nauseous",
            "The pain is on both sides of my head"
        ]
        
        for message in chat_messages:
            print(f"\n👤 User: {message}")
            result = self.test_chat(message, conversation_history)
            
            if result:
                print(f"🤖 AI: {result.get('response', 'No response')}")
                
                # Update conversation history
                conversation_history.append({
                    'message': message,
                    'symptom_category': result.get('symptom_category'),
                    'severity': result.get('severity')
                })
        
        print("\n" + "="*60)
        print("Demo Completed!")
        print("="*60)
        print("\nThe AI Symptom Checker is working correctly!")
        print("You can now integrate this API with your React Native app.")

def interactive_demo():
    """Interactive demo for manual testing"""
    demo = SymptomCheckerDemo()
    
    if not demo.test_health():
        return
    
    print("\n" + "="*60)
    print("Interactive Demo")
    print("="*60)
    print("Type 'quit' to exit")
    print("Type 'chat' to switch to chat mode")
    print("Type 'analyze' to switch to analysis mode")
    
    mode = "analyze"
    conversation_history = []
    
    while True:
        try:
            user_input = input(f"\n[{mode.upper()}] Enter your symptom: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'chat':
                mode = "chat"
                conversation_history = []
                print("Switched to chat mode")
                continue
            elif user_input.lower() == 'analyze':
                mode = "analyze"
                conversation_history = []
                print("Switched to analysis mode")
                continue
            elif not user_input:
                continue
            
            if mode == "analyze":
                result = demo.test_symptom_analysis(user_input)
                if result:
                    print(f"\n🤖 Analysis Results:")
                    print(f"   Category: {result.get('symptom_category')}")
                    print(f"   Severity: {result.get('severity')}")
                    print(f"   Confidence: {result.get('confidence'):.2f}")
                    print(f"   Follow-up Questions:")
                    for i, question in enumerate(result.get('follow_up_questions', [])[:3], 1):
                        print(f"     {i}. {question}")
                    print(f"   Recommendations:")
                    for i, rec in enumerate(result.get('recommendations', [])[:3], 1):
                        print(f"     {i}. {rec}")
            
            elif mode == "chat":
                result = demo.test_chat(user_input, conversation_history)
                if result:
                    print(f"\n🤖 AI: {result.get('response')}")
                    conversation_history.append({
                        'message': user_input,
                        'symptom_category': result.get('symptom_category'),
                        'severity': result.get('severity')
                    })
        
        except KeyboardInterrupt:
            print("\n\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Symptom Checker Demo')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run interactive demo')
    parser.add_argument('--url', default='http://localhost:5000',
                       help='API base URL')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo()
    else:
        demo = SymptomCheckerDemo(args.url)
        demo.run_demo()

if __name__ == "__main__":
    main() 