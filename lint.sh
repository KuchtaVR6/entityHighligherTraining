#!/bin/bash

# Track failures
FAILURE=0

echo "🚀 Running full lint & type-check pipeline..."

# Format with Black
echo "🧹 Formatting code with Black..."
if black src; then
  echo "✅ Black succeeded!"
else
  echo "❌ Black failed!"
  FAILURE=1
fi

# Lint check with Ruff
echo "🔎 Checking code with Ruff..."
if ruff check src; then
  echo "✅ Ruff check passed!"
else
  echo "❌ Ruff check failed!"
  FAILURE=1
fi

# Auto-fix with Ruff
echo "🧼 Auto-fixing with Ruff..."
if ruff check src --fix; then
  echo "✅ Ruff auto-fix completed!"
else
  echo "❌ Ruff auto-fix encountered issues!"
  FAILURE=1
fi

# Type-check with MyPy
echo "🔍 Type-checking with MyPy..."
if mypy src; then
  echo "✅ MyPy check passed!"
else
  echo "❌ MyPy check failed!"
  FAILURE=1
fi

# Final result
if [ $FAILURE -eq 0 ]; then
  echo "🎉 All checks passed successfully!"
  exit 0
else
  echo "🚨 Some checks failed. Please review the output."
