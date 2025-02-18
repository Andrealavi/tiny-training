#!/bin/bash

# Function to draw a line separator using the specified character
draw_separator() {
    local char=$1
    local width=$(tput cols)  # Get terminal width
    printf '%*s\n' "$width" | tr ' ' "$char"
}

# Function to display a centered title
display_title() {
    local title=$1
    local width=$(tput cols)
    local padding=$(( (width - ${#title}) / 2 ))
    printf "%${padding}s%s\n" "" "$title"
}

# Function to clear screen and show file content
display_file() {
    local file=$1
    clear  # Clear the terminal
    
    # Draw top border
    draw_separator "="
    
    # Display filename as title
    display_title "File: $file"
    
    # Draw separator under title
    draw_separator "-"
    
    # Display file content
    echo ""  # Add some spacing
    cat "$file"
    echo ""  # Add some spacing
    
    # Draw bottom border
    draw_separator "="
    
    # Display navigation instructions
    echo ""
    echo "Press [Enter] to view the next file, or [Ctrl+C] to exit"
}

# Main script
main() {
    # Check if directory argument is provided
    local dir=${1:-.}  # Use current directory if none specified
    
    # Get list of regular files in the directory
    local files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(find "$dir" -type f -print0)
    
    # Check if any files were found
    if [ ${#files[@]} -eq 0 ]; then
        echo "No files found in directory: $dir"
        exit 1
    fi
    
    # Display total number of files
    echo "Found ${#files[@]} files. Press [Enter] to start viewing..."
    read -r
    
    # Display files one by one
    for file in "${files[@]}"; do
        display_file "$file"
        read -r  # Wait for user input
    done
    
    # Display completion message
    clear
    echo "All files have been displayed."
}

# Run the script with provided directory or default to current directory
main "$@"
