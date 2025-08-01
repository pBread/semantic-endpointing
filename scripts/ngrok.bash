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
    ngrok "${args[@]}"
    return $?
}

# Main execution
main() {
    # Load environment variables
    load_env
    
    # Set defaults
    PORT="${PORT:-3333}"
    
    # Base arguments
    base_args=("http" "$PORT")
    exit_code=1  # 0 = success, 1 = error
    
    # Try legacy syntax if HOSTNAME is set
    if [[ -n "$HOSTNAME" ]]; then
        exec_ngrok "${base_args[@]}" "--hostname=$HOSTNAME"
        exit_code=$?
    fi
    
    # Try v3+ syntax if it fails and HOSTNAME is set
    if [[ -n "$HOSTNAME" && $exit_code -ne 0 ]]; then
        clear
        exec_ngrok "${base_args[@]}" "--url=$HOSTNAME"
        exit_code=$?
    fi
    
    # Clean up console and log error if both hostname attempts failed
    if [[ -n "$HOSTNAME" && $exit_code -ne 0 ]]; then
        clear
        echo "Ngrok unable to connect using HOSTNAME $HOSTNAME. These commands failed:" >&2
        echo -e "\t ngrok ${base_args[*]} --hostname=$HOSTNAME" >&2
        echo -e "\t ngrok ${base_args[*]} --url=$HOSTNAME" >&2
    fi
    
    # Start ngrok without hostname if hostname was invalid or undefined
    if [[ $exit_code -ne 0 ]]; then
        exec_ngrok "${base_args[@]}"
        exit_code=$?
    fi
    
    exit $exit_code
}

# Run main function
main "$@"