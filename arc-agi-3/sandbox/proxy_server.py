from __future__ import annotations

import argparse
import asyncio
import time
import urllib.parse
from pathlib import Path

_HEAD_LIMIT = 32 * 1024
_ALLOWED_PORT = 443


async def _dial(host: str, port: int, upstream: "str | None"):
    if not upstream:
        return await asyncio.open_connection(host, port)
    u = urllib.parse.urlparse(upstream)
    r, w = await asyncio.open_connection(u.hostname, u.port)
    w.write(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
    await w.drain()
    head = await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), timeout=30)
    status = head.split(b"\r\n", 1)[0].split()
    if len(status) < 2 or not status[1].startswith(b"200"):
        w.close()
        raise OSError(f"upstream proxy refused CONNECT {host}:{port}: {head[:120]!r}")
    return r, w


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} proxy: {msg}", flush=True)


def host_allowed(host: str, allow: list[str]) -> bool:
    h = host.lower().rstrip(".")
    for entry in allow:
        e = entry.lower().strip().rstrip(".")
        if not e:
            continue
        if e.startswith("."):
            if h == e[1:] or h.endswith(e):
                return True
        elif h == e:
            return True
    return False


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


async def _deny(writer: asyncio.StreamWriter, reason: str) -> None:
    _log(f"DENY {reason}")
    try:
        writer.write(
            b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
    except Exception:
        pass
    writer.close()


def make_handler(allow: list[str], upstream: "str | None" = None):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
        except Exception:
            writer.close()
            return
        line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        parts = line.split()
        if len(parts) != 3 or parts[0].upper() != "CONNECT":
            await _deny(writer, f"non-CONNECT request: {line[:120]!r}")
            return
        target = parts[1]
        host, _, port_s = target.rpartition(":")
        if not host:
            host, port_s = target, ""
        host = host.strip("[]")
        if port_s != str(_ALLOWED_PORT):
            await _deny(writer, f"CONNECT {target} (port not {_ALLOWED_PORT})")
            return
        if not host_allowed(host, allow):
            await _deny(writer, f"CONNECT {target} (host not in allowlist)")
            return
        try:
            up_reader, up_writer = await asyncio.wait_for(
                _dial(host, _ALLOWED_PORT, upstream), timeout=30
            )
        except Exception as e:
            _log(f"UPSTREAM FAIL {host}: {type(e).__name__}: {e}")
            try:
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
            except Exception:
                pass
            writer.close()
            return
        _log(f"ALLOW {host}:{_ALLOWED_PORT}")
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        await asyncio.gather(_pump(reader, up_writer), _pump(up_reader, writer))

    return handle


async def serve(args) -> None:
    handler = make_handler(args.allow, args.upstream)
    if args.socket is not None:
        args.socket.parent.mkdir(parents=True, exist_ok=True)
        try:
            args.socket.unlink()
        except OSError:
            pass
        server = await asyncio.start_unix_server(handler, path=str(args.socket), limit=_HEAD_LIMIT)
        where = str(args.socket)
    else:
        server = await asyncio.start_server(handler, "0.0.0.0", args.tcp_port, limit=_HEAD_LIMIT)
        where = f"0.0.0.0:{args.tcp_port}"
    _log(f"listening on {where}; allow: {', '.join(args.allow)}")
    async with server:
        await server.serve_forever()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--socket", type=Path, default=None)
    p.add_argument("--tcp-port", type=int, default=None)
    p.add_argument("--allow", required=True)
    p.add_argument("--upstream", default=None, metavar="URL")
    args = p.parse_args()
    if (args.socket is None) == (args.tcp_port is None):
        p.error("exactly one of --socket / --tcp-port is required")
    args.allow = [d for d in (s.strip() for s in args.allow.split(",")) if d]
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
