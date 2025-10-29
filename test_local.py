#!/usr/bin/env python3
"""
Simple test script to test the LLM locally without FastAPI
"""

from main import chain

def test_model():
    # Sample spending data for testing
    sample_spending = """
    Income: $5000/month
    Rent: $1500
    Groceries: $600
    Restaurants: $400
    Entertainment: $300
    Utilities: $200
    Transportation: $300
    Shopping: $500
    """
    
    print("Testing LLM with sample data...\n")
    print("Input:")
    print(sample_spending)
    print("\n" + "="*80 + "\n")
    print("Model Response:")
    print("="*80)
    
    try:
        # Invoke the chain with the sample data
        response = chain.invoke({"spending_data": sample_spending})
        print(response)
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Ollama installed and running")
        print("2. The 'deepseek-r1:8b' model installed (run: ollama pull deepseek-r1:8b)")
        return

def interactive_test():
    """Allow user to input their own prompt interactively"""
    print("Enter your spending data (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    
    spending_data = "\n".join(lines[:-1])  # Remove last empty line
    
    print("\n" + "="*80 + "\n")
    print("Model Response:")
    print("="*80)
    
    try:
        response = chain.invoke({"spending_data": spending_data})
        print(response)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        test_model()
        print("\n" + "="*80)
        print("\nTip: Run with --interactive flag to test with your own data:")
        print("  python test_local.py --interactive")

