## FileSystemAgent
The FileSystemAgent is an expert specialized in supporting with basic file handling operations. It can be used for various tasks such as:

1. **Directory Operations**: Change directory (cd), list directory contents (ls), create directory (mkdir), remove directory (rmdir), and their options.
2. **File Operations**: Read, write, append, create new files, delete files.
3. **Formatting Options**: Supports formatting options (e.g., JSON indent).

### Usage Scenarios
- "Create a new directory for a project called 'ProjectX' in the current working directory and create a README file inside it called 'README.md'."
- "Read the contents of a configuration file named 'config.json' and write the output to a new file called 'output.txt'."
- "List all files in the current directory and save the output to a file called 'file_list.txt'."
- "Read the json file 'data.json', apply JSON indentation, and write the formatted output to 'formatted_data.json'."

### Edge Cases
- **Naming conflict**: If a directory or file with the same name already exists, the agent will raise an error.
- **Permissiong issues**: The agent can only access directories and files for which it has the necessary permissions. If it tries to access a directory or file without permission, it will raise an error.
- **Non-existent directories or files**: If the agent tries to access a directory or file that does not exist, it will raise an error.

### Error Handling
- The agent will return specific error messages for common issues such as file not found, permission denied, etc.

### Performance Metrics
- The agent can handle up to 1000 file operations per minute under normal conditions.