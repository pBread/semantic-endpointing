#!/bin/bash

# Load environment variables from .env file
load_env() {
    if [[ -f ".env" ]]; then
        # Export variables from .env file, ignoring comments and empty lines
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Skip empty lines and comments
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
            
            # Export the variable if it's in KEY=VALUE format
            if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
                export "${BASH_REMATCH[1]}"="${BASH_REMATCH[2]}"
            fi
        done < ".env"
    fi
}

# Function to execute ngrok and return exit code
exec_ngrok() {
    local args=("$@")
    echo "Running: ngrok ${args[*]}"
    ngrok "${args[@]}"
    return $?
}

# Main execution
main() {
    # Load environment variables
    load_env
    
    # Set defaults
    PORT="${PORT:-8080}"
    
    # Base arguments
    base_args=("http" "$PORT")
    
    # Try with custom hostname first if HOSTNAME is set
    if [[ -n "$HOSTNAME" ]]; then
        echo "Attempting to start ngrok with hostname: $HOSTNAME"
        if exec_ngrok "${base_args[@]}" "--url=$HOSTNAME"; then
            exit 0
        else
            echo "Ngrok failed to connect using HOSTNAME $HOSTNAME" >&2
            echo "Falling back to random hostname..." >&2
            sleep 2
        fi
    fi
    
    # Start ngrok without custom hostname
    echo "Starting ngrok on port $PORT with random hostname..."
    exec_ngrok "${base_args[@]}"
    exit $?
}

# Run main function
main "$@"