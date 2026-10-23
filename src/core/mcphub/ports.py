"""Port allocation for built-in connectors.

The hub — not the connector module — owns the listen port: a stable
preferred port is derived from the connector id, probed for availability,
then handed to the child process as ``--port``. The auto-range bounds are
read through :mod:`mcphub.constants` at call time.
"""

import hashlib
import socket
from typing import Optional, Set

from . import constants


def preferred_port(connector_id: str) -> int:
    """Deterministic preferred port for a connector id inside the auto range.

    Hashes with blake2b, never ``hash()``: Python randomises
    ``PYTHONHASHSEED``, so ``hash()`` would reshuffle the ports at every
    app start.
    """
    digest = hashlib.blake2b(connector_id.encode("utf-8"), digest_size=4).hexdigest()
    return constants.AUTO_PORT_BASE + int(digest, 16) % constants.AUTO_PORT_SPAN


def port_is_free(port: int, claimed: Set[int]) -> bool:
    """True when the port is unclaimed by this hub run and bindable locally."""
    if port in claimed:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def allocate_port(connector_id: str, claimed: Set[int]) -> Optional[int]:
    """Probe forward from the preferred port (wrapping) until one is free.

    Returns the port after adding it to ``claimed`` so connectors of this
    run never collide with each other, or ``None`` when the whole range is
    occupied (foreign processes included).
    """
    preferred = preferred_port(connector_id)
    for offset in range(constants.AUTO_PORT_SPAN):
        port = (
            constants.AUTO_PORT_BASE
            + (preferred - constants.AUTO_PORT_BASE + offset) % constants.AUTO_PORT_SPAN
        )
        if port_is_free(port, claimed):
            claimed.add(port)
            return port
    return None
