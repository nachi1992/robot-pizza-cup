"""Station connection logging that distinguishes intentional close from failure.

Import after the existing Station Python environment has been configured.
"""
import logging

from station_py import new_station_client as _new_station_client


class _ConnectionLogger(logging.LoggerAdapter):
    def __init__(self, logger):
        super().__init__(logger, {})
        self.intentional_close = False

    def log(self, level, message, *args, **kwargs):
        # The upstream reader logs EOF as ERROR even when we closed its socket.
        # This adapter belongs to one connection; other clients keep their errors.
        if self.intentional_close and str(message).startswith('Connection lost while reading:'):
            level = logging.DEBUG
        super().log(level, message, *args, **kwargs)


async def new_station_client(server, logger):
    return await _new_station_client(server, _ConnectionLogger(logger))


def close_station_client(client):
    if isinstance(client.logger, _ConnectionLogger):
        client.logger.intentional_close = True
    client.close_connection()
