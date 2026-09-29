#!/usr/bin/env python3
"""Round-robin reverse proxy for DP sweep replicas.

One config = N independent TP=1 SGLang servers (one per GPU); benchmark clients only take a
single base_url, so this proxy fans their requests out round-robin. Requests are stateless
OpenAI-compat completions, so RR is safe.

Usage: rr_proxy.py <listen_port> <upstream_base> [<upstream_base> ...]
e.g.   rr_proxy.py 30050 http://localhost:30051 http://localhost:30052 ...
"""
import itertools
import os
import sys

from aiohttp import ClientSession, ClientTimeout, web

LISTEN_PORT = int(sys.argv[1])
UPSTREAMS = sys.argv[2:]
assert UPSTREAMS, "need at least one upstream"
_rr = itertools.count()

_session = None


async def _get_session():
    global _session
    if _session is None:
        # Default 7200s, matching config.sh's REASON_TIMEOUT: with a shorter proxy-side cap
        # a long thinking-mode generation gets severed HERE even though the client would
        # have kept waiting -- the partial generation is thrown away and retried from
        # scratch (measured ~40 min of restart waste on one 30-doc AIME seed at the old
        # 1800s default). Override via RR_PROXY_TIMEOUT (seconds).
        _timeout = float(os.environ.get("RR_PROXY_TIMEOUT", "7200"))
        _session = ClientSession(timeout=ClientTimeout(total=_timeout))
    return _session


async def handle(request: web.Request) -> web.Response:
    body = await request.read()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }
    sess = await _get_session()
    # Failover: a dead replica (e.g. one server OOM-killed mid-sweep) must not eat 1/N of
    # all requests forever -- on CONNECTION errors, walk the ring to the next upstream.
    # HTTP error responses are passed through as-is (they mean the server is alive).
    last_exc = None
    for attempt in range(len(UPSTREAMS)):
        upstream = UPSTREAMS[next(_rr) % len(UPSTREAMS)]
        url = upstream + str(request.rel_url)
        try:
            async with sess.request(request.method, url, data=body, headers=headers) as r:
                out = await r.read()
                ct = r.headers.get("Content-Type", "application/json")
                return web.Response(status=r.status, body=out, content_type=ct.split(";")[0])
        except Exception as e:  # ClientConnectorError, ServerDisconnected, timeouts
            last_exc = e
            continue
    return web.Response(status=502, text=f"all upstreams failed: {last_exc}")


app = web.Application(client_max_size=64 * 1024 * 1024)
app.router.add_route("*", "/{tail:.*}", handle)
web.run_app(app, host="127.0.0.1", port=LISTEN_PORT, print=lambda *_: None)
