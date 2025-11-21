#!/usr/bin/env python3
"""Test runner script for CVLab-Kit web helper
Runs both backend and frontend tests and generates coverage reports
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(
    cmd: list, cwd: Path = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)} in {cwd or Path.cwd()}")
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=check, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        if check:
            sys.exit(1)
        return e


def run_backend_tests(coverage: bool = True, verbose: bool = False):
    """Run backend tests with pytest"""
    print("=" * 50)
    print("Running Backend Tests")
    print("=" * 50)

    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(
            [
                "--cov=backend",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",
            ]
        )

    cmd.extend(["backend/tests/", "--tb=short"])

    result = run_command(cmd, cwd=Path("web_helper"), check=False)

    if result.returncode == 0:
        print("✅ Backend tests passed!")
        if coverage:
            print("📊 Coverage report generated in web_helper/htmlcov/")
    else:
        print("❌ Backend tests failed!")
        return False

    return True


def run_frontend_tests(coverage: bool = True, verbose: bool = False):
    """Run frontend tests with Vitest"""
    print("=" * 50)
    print("Running Frontend Tests")
    print("=" * 50)

    # Check if node_modules exists
    frontend_dir = Path("web_helper/frontend")
    if not (frontend_dir / "node_modules").exists():
        print("Installing frontend dependencies...")
        result = run_command(["npm", "install"], cwd=frontend_dir)
        if result.returncode != 0:
            print("❌ Failed to install frontend dependencies!")
            return False

    cmd = ["npm", "run"]

    if coverage:
        cmd.append("test:coverage")
    else:
        cmd.append("test")

    if not verbose:
        cmd.append("--reporter=basic")

    result = run_command(cmd, cwd=frontend_dir, check=False)

    if result.returncode == 0:
        print("✅ Frontend tests passed!")
        if coverage:
            print(
                "📊 Frontend coverage report generated in web_helper/frontend/coverage/"
            )
    else:
        print("❌ Frontend tests failed!")
        return False

    return True


def run_linting():
    """Run linting for both backend and frontend"""
    print("=" * 50)
    print("Running Code Quality Checks")
    print("=" * 50)

    success = True

    # Backend linting with ruff
    print("Checking backend code with ruff...")
    result = run_command(
        ["python", "-m", "ruff", "check", "backend/"],
        cwd=Path("web_helper"),
        check=False,
    )
    if result.returncode != 0:
        print("❌ Backend linting failed!")
        success = False
    else:
        print("✅ Backend linting passed!")

    # Frontend linting with ESLint
    print("Checking frontend code with ESLint...")
    result = run_command(
        ["npm", "run", "lint"], cwd=Path("web_helper/frontend"), check=False
    )
    if result.returncode != 0:
        print("❌ Frontend linting failed!")
        success = False
    else:
        print("✅ Frontend linting passed!")

    return success


def run_type_checking():
    """Run type checking for both backend and frontend"""
    print("=" * 50)
    print("Running Type Checking")
    print("=" * 50)

    success = True

    # Backend type checking with mypy
    print("Checking backend types with mypy...")
    try:
        result = run_command(
            ["python", "-m", "mypy", "backend/", "--ignore-missing-imports"],
            cwd=Path("web_helper"),
            check=False,
        )
        if result.returncode != 0:
            print("❌ Backend type checking failed!")
            success = False
        else:
            print("✅ Backend type checking passed!")
    except FileNotFoundError:
        print("⚠️ mypy not found, skipping backend type checking")

    # Frontend type checking with TypeScript
    print("Checking frontend types with TypeScript...")
    result = run_command(
        ["npx", "tsc", "--noEmit"], cwd=Path("web_helper/frontend"), check=False
    )
    if result.returncode != 0:
        print("❌ Frontend type checking failed!")
        success = False
    else:
        print("✅ Frontend type checking passed!")

    return success


def run_build_tests():
    """Test that both backend and frontend can build successfully"""
    print("=" * 50)
    print("Running Build Tests")
    print("=" * 50)

    success = True

    # Test frontend build
    print("Testing frontend build...")
    result = run_command(
        ["npm", "run", "build"], cwd=Path("web_helper/frontend"), check=False
    )
    if result.returncode != 0:
        print("❌ Frontend build failed!")
        success = False
    else:
        print("✅ Frontend build succeeded!")

    return success


def main():
    parser = argparse.ArgumentParser(description="Run CVLab-Kit web helper tests")
    parser.add_argument(
        "--no-coverage", action="store_true", help="Skip coverage reports"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--backend-only", action="store_true", help="Run only backend tests"
    )
    parser.add_argument(
        "--frontend-only", action="store_true", help="Run only frontend tests"
    )
    parser.add_argument("--no-lint", action="store_true", help="Skip linting")
    parser.add_argument("--no-types", action="store_true", help="Skip type checking")
    parser.add_argument("--no-build", action="store_true", help="Skip build tests")

    args = parser.parse_args()

    coverage = not args.no_coverage
    verbose = args.verbose

    all_passed = True
    run_count = 0

    # Change to project root
    os.chdir(Path(__file__).parent.parent)

    print("🧪 CVLab-Kit Web Helper Test Suite")
    print(f"Working directory: {Path.cwd()}")
    print()

    # Run linting
    if not args.no_lint:
        if not run_linting():
            all_passed = False
        run_count += 1
        print()

    # Run type checking
    if not args.no_types:
        if not run_type_checking():
            all_passed = False
        run_count += 1
        print()

    # Run backend tests
    if not args.frontend_only:
        if not run_backend_tests(coverage=coverage, verbose=verbose):
            all_passed = False
        run_count += 1
        print()

    # Run frontend tests
    if not args.backend_only:
        if not run_frontend_tests(coverage=coverage, verbose=verbose):
            all_passed = False
        run_count += 1
        print()

    # Run build tests
    if not args.no_build:
        if not run_build_tests():
            all_passed = False
        run_count += 1
        print()

    # Summary
    print("=" * 50)
    print("Test Summary")
    print("=" * 50)

    if all_passed:
        print(f"🎉 All tests passed! ({run_count} test suites)")
        print()
        if coverage:
            print("📊 Coverage reports:")
            print("  - Backend: web_helper/htmlcov/index.html")
            print("  - Frontend: web_helper/frontend/coverage/index.html")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
