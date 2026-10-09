"""Operations and client integration for a local memory-os-v4 ALPHA deployment.

Everything here talks to the shared HTTP server as a *client* (never opens the store for writing),
except ``backup`` which copies the store while the service is stopped. Paths are configurable through
environment variables (see :mod:`memory_os_v4.ops.settings`); nothing is specific to one machine.
"""
