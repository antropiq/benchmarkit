#!/usr/bin/env python3
"""
Console MCP Server - Allows LLMs to execute console commands via MCP protocol.

This server exposes tools that enable AI models to run shell commands on the local system.
It implements the Model Context Protocol (MCP) and serves at http://localhost:9999/mcp

Usage:
    python -m mcp_connectors.console_mcp               # Start server on default port 9999
    python -m mcp_connectors.console_mcp --port 8080   # Start server on custom port
"""

import argparse
import subprocess
import shlex
from typing import Optional
from mcp.server.mcpserver import MCPServer

# Create MCP server instance
# streamable_http_path="/mcp" is the default, which matches your requirement
server = MCPServer(
    name="console-mcp",
    instructions="A server that allows LLMs to execute console commands on the local system. "
                 "Useful for file operations, system administration, and task automation.",
)

# Define allowed commands (security whitelist)
ALLOWED_COMMANDS = {
    "ls", "dir", "cat", "echo", "pwd", "whoami", "date", "time",
    "find", "grep", "sed", "wc", "head", "tail", "sort", "uniq", "cut",
    "mkdir", "touch", "cp", "mv", "rm", "chmod", "chown",
    "ps", "top", "df", "du", "free", "uptime", "hostname",
    "curl", "wget", "ping", "nslookup", "dig",
    "python3", "python", "node", "npm", "pip", "git",
    "vim", "nano", "less", "more", "man",
    "env", "printenv", "id", "groups", "who", "w", "actutopdf.py"
}

# Commands that are always blocked (security)
BLOCKED_COMMANDS = {
    "rm", "rmdir", "chmod", "chown", "mkfs", "dd", "shutdown", "reboot", "poweroff"
}


def validate_command(command: str) -> tuple[bool, str]:
    """
    Validate if a command is safe to execute.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not command or not command.strip():
        return False, "Command cannot be empty"

    # Parse the command to get the base command
    parts = shlex.split(command)
    if not parts:
        return False, "Invalid command format"

    base_command = parts[0]

    # Check for blocked commands
    if base_command in BLOCKED_COMMANDS:
        return False, f"Command '{base_command}' is blocked for security reasons"

    # Check if command is in allowed list
    if base_command not in ALLOWED_COMMANDS:
        return False, f"Command '{base_command}' is not in the allowed commands list"

    # Additional safety checks
    dangerous_patterns = ["|", ";", "&", ">", "<", "$(", "`", "||", "&&"]
    for pattern in dangerous_patterns:
        if pattern in command:
            return False, f"Command contains dangerous pattern '{pattern}'"

    return True, ""


@server.tool()
def execute_command(
    command: str,
    timeout: int = 30,
    cwd: Optional[str] = None
) -> str:
    """
    Execute a console command and return the output.

    Args:
        command: The shell command to execute (e.g., "ls -la", "cat /etc/hostname")
        timeout: Maximum execution time in seconds (default: 30)
        cwd: Working directory for command execution (optional)

    Returns:
        Command output as a string, or error message if execution fails

    Example:
        execute_command("ls -la /tmp", timeout=10)
    """
    # Validate the command
    is_valid, error_msg = validate_command(command)
    if not is_valid:
        return f"Error: {error_msg}"

    try:
        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )

        # Build response
        output_parts = []

        if result.stdout:
            output_parts.append(f"STDOUT:\n{result.stdout}")

        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")

        if result.returncode != 0:
            output_parts.append(f"\nExit Code: {result.returncode}")

        if not output_parts:
            output_parts.append("Command executed successfully (no output)")

        return "\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except FileNotFoundError:
        return f"Error: Command not found"
    except PermissionError:
        return f"Error: Permission denied"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@server.tool()
def list_allowed_commands() -> str:
    """
    List all commands that are allowed to be executed by this server.

    Returns:
        Formatted list of allowed commands
    """
    sorted_commands = sorted(ALLOWED_COMMANDS)
    return f"Allowed commands ({len(sorted_commands)} total):\n" + \
           "\n".join(f"  - {cmd}" for cmd in sorted_commands)


@server.tool()
def check_command_allowed(command: str) -> str:
    """
    Check if a specific command is allowed to be executed.

    Args:
        command: The command to check

    Returns:
        Whether the command is allowed and any restrictions
    """
    is_valid, error_msg = validate_command(command)
    if is_valid:
        return f"Command '{command}' is ALLOWED for execution"
    else:
        return f"Command '{command}' is NOT ALLOWED: {error_msg}"


@server.tool()
def safe_file_read(filepath: str, max_lines: int = 1000) -> str:
    """
    Safely read a file with line limit protection.

    Args:
        filepath: Path to the file to read
        max_lines: Maximum number of lines to read (default: 1000)

    Returns:
        File contents or error message

    Example:
        safe_file_read("/etc/hostname", max_lines=10)
    """
    # Validate filepath
    if not filepath or ".." in filepath:
        return "Error: Invalid filepath"

    # Use cat with head to enforce line limit
    command = f"head -n {max_lines} {shlex.quote(filepath)}"
    is_valid, error_msg = validate_command(command)

    if not is_valid:
        return f"Error: {error_msg}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )

        output = result.stdout
        if result.returncode != 0:
            return f"Error reading file: {result.stderr}"

        # Check if we hit the limit
        line_count = len(output.split('\n'))
        if line_count >= max_lines:
            output += f"\n... (output truncated at {max_lines} lines)"

        return f"File: {filepath}\n{output}"

    except Exception as e:
        return f"Error: {str(e)}"


@server.tool()
def safe_file_write(filepath: str, content: str) -> str:
    """
    Safely write content to a file.

    Args:
        filepath: Path to the file to write
        content: Content to write to the file

    Returns:
        Success or error message

    Example:
        safe_file_write("/tmp/test.txt", "Hello, World!")
    """
    # Validate filepath
    if not filepath or ".." in filepath:
        return "Error: Invalid filepath"

    try:
        with open(filepath, 'w') as f:
            f.write(content)

        return f"Successfully wrote {len(content)} bytes to {filepath}"

    except PermissionError:
        return f"Error: Permission denied writing to {filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Console MCP Server - Execute console commands via MCP protocol"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9999,
        help="Port to listen on (default: 9999)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    if args.debug:
        server.settings.log_level = "DEBUG"

    print(f"🚀 Console MCP Server starting...")
    print(f"   Endpoint: http://{args.host}:{args.port}/mcp")
    print(f"   Allowed commands: {len(ALLOWED_COMMANDS)}")
    print(f"   Blocked commands: {len(BLOCKED_COMMANDS)}")
    print(f"\n   Press Ctrl+C to stop\n")

    # Run the server with streamable HTTP transport
    try:
        server.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
        )
    except (KeyboardInterrupt, SystemExit):
        print("\nConsole MCP Server stopped.")


if __name__ == "__main__":
    main()
