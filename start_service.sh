#!/bin/sh

# Wait for the Ollama service to be available
while ! wget -q -O- http://${OLLAMA_HOST}:${OLLAMA_PORT} > /dev/null 2>&1; do
  echo "Waiting for Ollama service to start..."
  sleep 2
done

# Get the actual container name
CONTAINER_NAME=$(docker ps --filter "name=ollama-container" --format "{{.Names}}")

# Execute the ollama pull llama3 command in the Ollama container
docker exec $CONTAINER_NAME ollama pull llama3
