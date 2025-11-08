#!/usr/bin/env python3
"""
Simple test script to test the LLM locally without FastAPI
"""

from main import chain

def test_model():
    # Sample spending data for testing
    sample_spending = """
    "added": [
    {
      "account_id": "BxBXxLj1m4HMXBm9WZZmCWVbPjX16EHwv99vp",
      "account_owner": null,
      "amount": 72.1,
      "iso_currency_code": "USD",
      "unofficial_currency_code": null,
      "check_number": null,
      "counterparties": [
        {
          "name": "Walmart",
          "type": "merchant",
          "logo_url": "https://plaid-merchant-logos.plaid.com/walmart_1100.png",
          "website": "walmart.com",
          "entity_id": "O5W5j4dN9OR3E6ypQmjdkWZZRoXEzVMz2ByWM",
          "confidence_level": "VERY_HIGH"
        }
      ],
      "date": "2023-09-24",
      "datetime": "2023-09-24T11:01:01Z",
      "authorized_date": "2023-09-22",
      "authorized_datetime": "2023-09-22T10:34:50Z",
      "location": {
        "address": "13425 Community Rd",
        "city": "Poway",
        "region": "CA",
        "postal_code": "92064",
        "country": "US",
        "lat": 32.959068,
        "lon": -117.037666,
        "store_number": "1700"
      },
      "name": "PURCHASE WM SUPERCENTER #1700",
      "merchant_name": "Walmart",
      "merchant_entity_id": "O5W5j4dN9OR3E6ypQmjdkWZZRoXEzVMz2ByWM",
      "logo_url": "https://plaid-merchant-logos.plaid.com/walmart_1100.png",
      "website": "walmart.com",
      "payment_meta": {
        "by_order_of": null,
        "payee": null,
        "payer": null,
        "payment_method": null,
        "payment_processor": null,
        "ppd_id": null,
        "reason": null,
        "reference_number": null
      },
      "payment_channel": "in store",
      "pending": false,
      "pending_transaction_id": "no86Eox18VHMvaOVL7gPUM9ap3aR1LsAVZ5nc",
      "personal_finance_category": {
        "primary": "GENERAL_MERCHANDISE",
        "detailed": "GENERAL_MERCHANDISE_SUPERSTORES",
        "confidence_level": "VERY_HIGH"
      },
      "personal_finance_category_icon_url": "https://plaid-category-icons.plaid.com/PFC_GENERAL_MERCHANDISE.png",
      "transaction_id": "lPNjeW1nR6CDn5okmGQ6hEpMo4lLNoSrzqDje",
      "transaction_code": null,
      "transaction_type": "place"
    }
  ],
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

