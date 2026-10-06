from __future__ import annotations

import asyncio
import sys


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


def make_handler(sock_path: str):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            up_reader, up_writer = await asyncio.open_unix_connection(sock_path)
        except Exception as e:
            print(f"bridge: cannot reach {sock_path}: {e}", file=sys.stderr, flush=True)
            writer.close()
            return
        await asyncio.gather(_pump(reader, up_writer), _pump(up_reader, writer))

    return handle


async def main(specs: list[str]) -> None:
    servers = []
    for spec in specs:
        port_s, _, path = spec.partition(":")
        server = await asyncio.start_server(make_handler(path), "127.0.0.1", int(port_s))
        servers.append(server)
        print(f"bridge: 127.0.0.1:{port_s} -> {path}", flush=True)
    print("bridge: ready", flush=True)
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: jail_bridge.py PORT:SOCKET_PATH...", file=sys.stderr)
        raise SystemExit(2)
    try:
        asyncio.run(main(sys.argv[1:]))
    except KeyboardInterrupt:
        pass
