#!/usr/bin/env python3
"""
Console MCP Server - Allows LLMs to execute console commands via MCP protocol.

This server exposes tools that enable AI models to run shell commands on the local system.
It implements the Model Context Protocol (MCP) and serves at http://localhost:9999/mcp

Reading and editing files is the job of the fedit-mcp connector (line-numbered
reads, match-verified edits, backups and revert). Enable both together
(MCP_ENABLED=console_mcp,fedit_mcp): this server for shell commands, fedit-mcp
for any file content work.

Root scoping: the server MUST be started with --root-dir. The console starts
in that directory and the security policy is path-scoped: commands whose path
arguments all stay inside the root run WITHOUT any policy restrictions; the
block/allow lists and dangerous-pattern checks only apply to commands that
reference paths outside the root. Privilege/power/disk/service commands
(sudo, reboot, mkfs, systemctl, ...) are never path work and stay blocked
everywhere.

Usage:
    python -m mcp_connectors.console_mcp --root-dir /path/to/root              # default port 9999
    python -m mcp_connectors.console_mcp --root-dir /path --port 8080          # custom port
    python -m mcp_connectors.console_mcp --root-dir /path --transport stdio    # stdio (for MCP hubs)
"""

import argparse
import os
import subprocess
import shlex
import sys
from pathlib import Path
from typing import Optional
from mcp.server.mcpserver import MCPServer

# Root directory the console is scoped to; set by main() from --root-dir.
# When None (e.g. module imported without main()), the full policy applies to
# every command (legacy behaviour).
ROOT_DIR: Optional[Path] = None

# Create MCP server instance
# streamable_http_path="/mcp" is the default, which matches your requirement
server = MCPServer(
    name="console-mcp",
    instructions="A server that allows LLMs to execute console commands, scoped to a root "
                 "directory chosen at startup (--root-dir). Commands that only touch paths "
                 "inside that root run without policy restrictions; commands referencing "
                 "paths outside the root must pass the allow/block policy. "
                 "For reading or editing file contents, use the fedit-mcp connector instead "
                 "(read_file / preview_edit / apply_edit / revert_edit).",
)

# ---------------------------------------------------------------------------
# Security policy: ALLOWED is a whitelist, BLOCKED always wins.
# Keep the two lists disjoint -- entries in both are effectively blocked
# and only create confusion.
#
# The policy is PATH-SCOPED: it only applies to commands that reference a
# path outside ROOT_DIR. Commands whose path arguments all resolve inside
# the root bypass every check (the root is the operator's workspace).
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

# Privilege / power / disk / service commands are NOT path-scoped file work:
# they stay blocked even when a command would otherwise run inside the root.
ALWAYS_BLOCKED_COMMANDS = {
    "mkfs", "dd", "fdisk", "parted", "mount", "umount",
    "shutdown", "reboot", "poweroff", "halt",
    "sudo", "su", "doas", "runuser",
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


def resolve_as_path(token: str) -> Path:
    """Resolve a command token as a path; relative paths anchor at ROOT_DIR."""
    expanded = os.path.expanduser(token)
    path = Path(expanded)
    if not path.is_absolute():
        path = (ROOT_DIR or Path.cwd()) / path
    try:
        return path.resolve()
    except OSError:
        return Path(os.path.normpath(str(path)))


def is_within_root(token: str) -> bool:
    """True when the token resolves to a path inside (or equal to) ROOT_DIR."""
    if ROOT_DIR is None:
        return False
    resolved = resolve_as_path(token)
    return resolved == ROOT_DIR or ROOT_DIR in resolved.parents


def path_argument_tokens(parts: list[str]) -> list[str]:
    """Tokens of a parsed command that could name a filesystem path."""
    tokens: list[str] = []
    for token in parts[1:]:
        if token.startswith("-"):
            _, sep, value = token.partition("=")
            if sep and value.startswith(("/", "~", "./", "../")):
                tokens.append(value)
        else:
            tokens.append(token)
    return tokens


def validate_command(command: str) -> tuple[bool, str]:
    """
    Validate if a command is safe to execute.

    Policy is path-scoped: a command whose path arguments all stay inside
    ROOT_DIR bypasses every check; anything referencing a path outside the
    root must pass the block list, the allow list and the pattern checks.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not command or not command.strip():
        return False, "Command cannot be empty"

    # Parse the command to get the base command
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return False, f"Invalid command format: {exc}"
    if not parts:
        return False, "Invalid command format"

    base_command = parts[0]

    # Privilege / power / disk / service commands are blocked everywhere
    if base_command in ALWAYS_BLOCKED_COMMANDS:
        return False, f"Command '{base_command}' is blocked for security reasons"

    # Commands whose path arguments all stay inside the root bypass the policy
    path_tokens = path_argument_tokens(parts)
    if ROOT_DIR is not None and path_tokens and all(is_within_root(t) for t in path_tokens):
        return True, ""

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
        cwd: Working directory for command execution (optional, must stay
             inside the server's --root-dir; defaults to the root)

    Returns:
        Command output as a string, or error message if execution fails

    Example:
        execute_command("ls -la sub/dir", timeout=10)
    """
    if ROOT_DIR is None:
        return "Error: server is not rooted; start it with --root-dir"

    work_dir = str(ROOT_DIR)
    if cwd is not None:
        candidate = resolve_as_path(cwd)
        if candidate != ROOT_DIR and ROOT_DIR not in candidate.parents:
            return f"Error: cwd '{cwd}' is outside the server root directory"
        work_dir = str(candidate)

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
            cwd=work_dir
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
        "--root-dir",
        type=str,
        required=True,
        help="Root directory the console starts in and is scoped to (required). "
             "Commands touching only paths inside it bypass the security policy."
    )
    parser.add_argument(
        "--transport", "-t",
        type=str,
        choices=("stdio", "streamable-http"),
        default="streamable-http",
        help="Transport to serve on (default: streamable-http)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    global ROOT_DIR
    root = Path(os.path.expanduser(args.root_dir))
    if not root.is_dir():
        parser.error(f"--root-dir '{args.root_dir}' is not an existing directory")
    ROOT_DIR = root.resolve()

    if args.debug:
        server.settings.log_level = "DEBUG"

    # stdio mode carries JSON-RPC on stdout: keep the banner off it.
    banner = sys.stderr if args.transport == "stdio" else sys.stdout
    print(f"🚀 Console MCP Server starting...", file=banner)
    if args.transport == "stdio":
        print(f"   Transport: stdio", file=banner)
    else:
        print(f"   Endpoint: http://{args.host}:{args.port}/mcp", file=banner)
    print(f"   Root directory: {ROOT_DIR}", file=banner)
    print(f"   Allowed commands: {len(ALLOWED_COMMANDS)}", file=banner)
    print(f"   Blocked commands: {len(BLOCKED_COMMANDS)}", file=banner)
    print(f"\n   Press Ctrl+C to stop\n", file=banner)

    # Run the server with the selected transport
    try:
        if args.transport == "stdio":
            server.run(transport="stdio")
        else:
            server.run(
                transport="streamable-http",
                host=args.host,
                port=args.port,
            )
    except (KeyboardInterrupt, SystemExit):
        print("\nConsole MCP Server stopped.", file=banner)


if __name__ == "__main__":
    main()
