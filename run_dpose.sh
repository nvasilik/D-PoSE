#!/usr/bin/env bash
set -e

# --- CONFIGURATION ---
COMPOSE_FILE="docker-compose.yaml"
SERVICE_NAME="bedlam-depth"
CONTAINER_NAME="dposewebcam"

# --- CHECKS ---
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose plugin not found. Please install it (Docker Desktop or docker-compose-plugin)."
    exit 1
fi

# --- X11 FORWARDING SETUP ---
xhost +local:docker >/dev/null 2>&1 || true
export DISPLAY=${DISPLAY:-:0}

# --- HELPER FUNCTIONS ---
is_running() {
    docker ps --filter "name=${CONTAINER_NAME}" --filter "status=running" | grep -q "${CONTAINER_NAME}"
}

show_menu() {
    echo ""
    echo "============================="
    echo "   DPOSE ROS CONTAINER MENU"
    echo "============================="
    if is_running; then
        echo "✅ Container is RUNNING"
        echo "1) Attach to container shell"
        echo "2) Show logs"
        echo "3) Stop container"
        echo "4) Remove container"
        echo "5) Exit"
    else
        echo "❌ Container is NOT running"
        echo "1) Build image"
        echo "2) Start & Attach to container"
        echo "3) Exit"
    fi
    echo ""
}

while true; do
    show_menu
    if is_running; then
        read -rp "Select an option [1-5]: " choice
        case $choice in
            1)
                echo "🔗 Attaching to container shell..."
                docker exec -it "$CONTAINER_NAME" bash
                ;;
            2)
                echo "📜 Showing logs (Ctrl+C to exit)..."
                docker compose -f "$COMPOSE_FILE" logs -f "$SERVICE_NAME"
                ;;
            3)
                echo "🛑 Stopping container..."
                docker compose -f "$COMPOSE_FILE" stop "$SERVICE_NAME"
                ;;
            4)
                echo "⚠️ Removing container..."
                docker compose -f "$COMPOSE_FILE" down
                ;;
            5)
                echo "👋 Exiting."
                exit 0
                ;;
            *)
                echo "❌ Invalid option."
                ;;
        esac
    else
        read -rp "Select an option [1-3]: " choice
        case $choice in
            1)
                echo "🔨 Building Docker image..."
                docker compose -f "$COMPOSE_FILE" build "$SERVICE_NAME"
                ;;
            2)
                echo "🚀 Starting container..."
                docker compose -f "$COMPOSE_FILE" up -d "$SERVICE_NAME"
                echo "🔗 Attaching to container shell..."
                docker exec -it "$CONTAINER_NAME" bash
                ;;
            3)
                echo "👋 Exiting."
                exit 0
                ;;
            *)
                echo "❌ Invalid option."
                ;;
        esac
    fi
done
