#!/usr/bin/env python3
"""
Console MCP Server - Allows LLMs to execute console commands via MCP protocol.

This server exposes tools that enable AI models to run shell commands on the local system.
It implements the Model Context Protocol (MCP) and serves at http://localhost:9999/mcp

Reading and editing files is the job of the fedit-mcp connector (line-numbered
reads, match-verified edits, backups and revert). Enable both together
(MCP_ENABLED=console_mcp,fedit_mcp): this server for shell commands, fedit-mcp
for any file content work.

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
                 "Useful for system inspection, archives, checksums, and task automation. "
                 "For reading or editing file contents, use the fedit-mcp connector instead "
                 "(read_file / preview_edit / apply_edit / revert_edit).",
)

# ---------------------------------------------------------------------------
# Security policy: ALLOWED is a whitelist, BLOCKED always wins.
# Keep the two lists disjoint -- entries in both are effectively blocked
# and only create confusion.
# ---------------------------------------------------------------------------

# Commands that are always blocked (checked first, security)
BLOCKED_COMMANDS = {
    # destructive file operations
    "rm", "rmdir", "shred", "truncate",
    # permissions / ownership
    "chmod", "chown",
    # disk / system destruction
    "mkfs", "dd", "fdisk", "parted", "mount", "umount",
    # power state
    "shutdown", "reboot", "poweroff", "halt",
    # privilege escalation
    "sudo", "su", "doas", "runuser",
    # package / service control
    "pacman", "systemctl", "crontab",
}

# Define allowed commands (security whitelist)
ALLOWED_COMMANDS = {
    # --- basic file & text (read-only) ---
    "ls", "dir", "cat", "echo", "pwd", "whoami", "date", "time",
    "head", "tail", "wc", "sort", "uniq", "cut",
    "grep", "sed", "awk", "tr", "tac", "nl", "column", "comm", "paste",
    "less", "more", "man", "which", "file", "stat",

    # --- paths & paths metadata ---
    "find", "basename", "dirname", "realpath", "readlink",
    "diff", "cmp", "seq", "printf", "bc", "numfmt",

    # --- benign file creation (destructive ops stay blocked) ---
    "mkdir", "touch", "cp", "mv", "ln", "mktemp",

    # --- checksums & archives ---
    "md5sum", "sha1sum", "sha256sum",
    "tar", "gzip", "unzip", "zip",

    # --- system visibility (read-only) ---
    "ps", "top", "vmstat", "df", "du", "free", "uptime", "hostname",
    "uname", "lscpu", "lsb_release", "nproc", "id", "groups", "who", "w",

    # --- network ---
    "curl", "wget", "ping", "nslookup", "dig", "ss", "ip", "tracepath",

    # --- environment ---
    "env", "printenv", "sleep",

    # --- dev tooling ---
    "python3", "python", "node", "npm", "pip", "git",
    "jq", "sqlite3", "openssl",

    # --- editors (kept from your original list) ---
    "vim", "nano",
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
