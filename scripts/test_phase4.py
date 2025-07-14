#!/usr/bin/env python3
"""
Quick test script for Phase 4 components

Tests basic functionality of:
- API endpoints
- Dashboard components  
- Test suite
"""

import sys
import subprocess
import time
import requests
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # API imports
        from src.app.api import app
        print("✅ API module imported successfully")
        
        # Dashboard imports  
        import streamlit
        import plotly
        print("✅ Dashboard dependencies imported successfully")
        
        # Test imports
        import pytest
        import fastapi
        print("✅ Test dependencies imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nPlease install missing dependencies:")
        print("pip install fastapi uvicorn streamlit plotly pytest")
        return False


def test_api_startup():
    """Test that API can start"""
    print("\nTesting API startup...")
    
    try:
        # Try importing and creating test client
        from fastapi.testclient import TestClient
        from src.app.api import app
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        if response.status_code == 200:
            print("✅ API health check passed")
            print(f"   Status: {response.json()['status']}")
            return True
        else:
            print(f"❌ API health check failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def test_models():
    """Test model functionality"""
    print("\nTesting model components...")
    
    try:
        from src.models.baseline import BaselineModel, LinearRegressionBaseline
        
        # Create a simple model
        model = LinearRegressionBaseline(random_state=42)
        print("✅ Model classes imported successfully")
        
        # Test basic functionality
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=50)
        data = pd.DataFrame({
            'Open': np.random.randn(50).cumsum() + 100,
            'High': np.random.randn(50).cumsum() + 101,
            'Low': np.random.randn(50).cumsum() + 99,
            'Close': np.random.randn(50).cumsum() + 100,
            'Volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
        
        # Test model operations
        model.fit(data)
        predictions = model.predict(data.tail(30))
        if len(predictions) == 30:
            print("✅ Model training and prediction successful")
            return True
        else:
            print("❌ Model prediction failed")
            return False
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_data_pipeline():
    """Test data pipeline components"""
    print("\nTesting data pipeline...")
    
    try:
        from src.features.indicators import calculate_moving_averages
        import pandas as pd
        import numpy as np
        
        # Create sample data
        data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 5000000, 100)
        })
        
        # Test feature engineering
        result = calculate_moving_averages(data)
        
        if 'SMA_10' in result.columns:
            print("✅ Feature engineering successful")
            return True
        else:
            print("❌ Feature engineering failed")
            return False
            
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        return False


def run_quick_tests():
    """Run pytest on a subset of tests"""
    print("\nRunning quick pytest suite...")
    
    try:
        # Run a quick subset of tests
        result = subprocess.run(
            ["pytest", "tests/test_api.py::TestHealthEndpoint", "-v", "-q"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Quick tests passed")
            return True
        else:
            print("❌ Some tests failed")
            print(result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ Could not run tests: {e}")
        return False


def main():
    """Run all Phase 4 tests"""
    print("=" * 50)
    print("Phase 4 Component Testing")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("API", test_api_startup()))
    results.append(("Models", test_models()))
    results.append(("Data Pipeline", test_data_pipeline()))
    
    # Only run pytest if other tests pass
    if all(r[1] for r in results):
        results.append(("Quick Tests", run_quick_tests()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<30} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All Phase 4 components are working correctly!")
        print("\nNext steps:")
        print("1. Start API: uvicorn src.app.api:app --reload")
        print("2. Start Dashboard: streamlit run src/app/dashboard.py")
        print("3. Run full tests: pytest tests/ -v")
    else:
        print("\n⚠️  Some components need attention. Check the errors above.")
        print("\nTry installing missing dependencies:")
        print("pip install -r requirements-core.txt")
        print("pip install fastapi uvicorn streamlit plotly psutil pytest")


if __name__ == "__main__":
    main()