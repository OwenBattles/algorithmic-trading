#!/usr/bin/env python3
"""
Test script to verify the algorithmic trading system setup.
Run this script to check if all dependencies and components are working correctly.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import matplotlib: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import scikit-learn: {e}")
        return False
    
    try:
        import joblib
        print("✓ joblib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import joblib: {e}")
        return False
    
    try:
        import flask
        print("✓ flask imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import flask: {e}")
        return False
    
    return True

def test_file_structure():
    """Test if the project structure is correct"""
    print("\nTesting project structure...")
    
    required_files = [
        "daily_trades.py",
        "graph.py",
        "requirements.txt",
        "README.md",
        "setup.py",
        ".gitignore"
    ]
    
    required_dirs = [
        "stock_notebooks",
        "stock_notebooks/models",
        "stock_notebooks/stock_data",
        "trading_view_app",
        "tutorial"
    ]
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_good = False
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}/ exists")
        else:
            print(f"✗ {directory}/ missing")
            all_good = False
    
    return all_good

def test_model_files():
    """Test if model files exist"""
    print("\nTesting model files...")
    
    model_files = [
        "stock_notebooks/models/AAPL_model.pkl",
        "stock_notebooks/models/AMZN_model.pkl",
        "stock_notebooks/models/KO_model.pkl",
        "stock_notebooks/models/MSFT_model.pkl"
    ]
    
    all_good = True
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✓ {model_file} exists")
        else:
            print(f"✗ {model_file} missing")
            all_good = False
    
    return all_good

def test_data_files():
    """Test if data files exist"""
    print("\nTesting data files...")
    
    data_files = [
        "stock_notebooks/stock_data/AAPL_price_data.csv",
        "stock_notebooks/stock_data/AMZN_price_data.csv",
        "stock_notebooks/stock_data/KO_price_data.csv",
        "stock_notebooks/stock_data/MSFT_price_data.csv"
    ]
    
    all_good = True
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"✓ {data_file} exists")
        else:
            print(f"✗ {data_file} missing")
            all_good = False
    
    return all_good

def test_python_files():
    """Test if Python files can be imported"""
    print("\nTesting Python file imports...")
    
    try:
        import daily_trades
        print("✓ daily_trades.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import daily_trades.py: {e}")
        return False
    
    try:
        import graph
        print("✓ graph.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import graph.py: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("ALGORITHMIC TRADING SYSTEM - SETUP TEST")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_file_structure),
        ("Model Files", test_model_files),
        ("Data Files", test_data_files),
        ("Python Files", test_python_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your algorithmic trading system is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python daily_trades.py' to start trading simulation")
        print("2. Run 'python graph.py' to visualize portfolio performance")
        print("3. Check the README.md for detailed usage instructions")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
