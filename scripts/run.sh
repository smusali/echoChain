#!/usr/bin/env bash

echo "Experiment: Redis Technical Telephone Game"
if python -m echochain --start "Redis uses single-threaded event loops with epoll for I/O multiplexing"; then
  echo "✅ Redis experiment completed successfully"
  echo ""
  echo "The system is working correctly!"
  echo "Check the experiments/ directory for detailed results."
else
  echo "❌ Redis experiment failed"
fi
