import sys
import pytest

if __name__ == "__main__":
    print("Starting tests...")
    # Add app to path
    import os
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Run pytest
    exit_code = pytest.main(sys.argv[1:])
    print(f"Tests finished with exit code {exit_code}")
    sys.exit(exit_code)
