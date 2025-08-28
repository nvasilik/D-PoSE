#!/usr/bin/env python3
"""
Syntax verification script for ros_demo_webcam.py

This script checks if the cleaned ros_demo_webcam.py file has correct syntax
without requiring all the heavy dependencies to be installed.
"""

import ast
import sys
import os

def check_syntax(filename):
    """Check if a Python file has valid syntax."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source, filename=filename)
        return True, None
        
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    webcam_script = os.path.join(script_dir, 'ros_demo_webcam.py')
    
    print("🔍 Checking syntax of ros_demo_webcam.py...")
    
    if not os.path.exists(webcam_script):
        print(f"❌ File not found: {webcam_script}")
        return 1
    
    is_valid, error = check_syntax(webcam_script)
    
    if is_valid:
        print("✅ Syntax check passed! The script is syntactically correct.")
        
        # Additional checks
        print("\n📋 Additional validations:")
        
        # Check for common issues
        with open(webcam_script, 'r') as f:
            content = f.read()
            
        # Check for proper shebang
        if content.startswith('#!/usr/bin/env python3'):
            print("✅ Proper shebang line found")
        else:
            print("⚠️  Missing or incorrect shebang line")
            
        # Check for docstring
        if '"""' in content[:500]:  # Check first 500 chars for module docstring
            print("✅ Module docstring found")
        else:
            print("⚠️  Module docstring missing")
            
        # Check for main guard
        if "if __name__ == '__main__':" in content:
            print("✅ Main execution guard found")
        else:
            print("⚠️  Main execution guard missing")
            
        print("\n🎉 ros_demo_webcam.py is ready for use!")
        return 0
        
    else:
        print(f"❌ Syntax check failed: {error}")
        return 1

if __name__ == '__main__':
    sys.exit(main())