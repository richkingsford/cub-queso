"""Shared helpers for livestream setup."""
import threading

from helper_stream_server import StreamServer, format_stream_url


def build_stream_provider(state, frame_key="frame", lock_key="lock"):
    def _provider():
        lock = state.get(lock_key)
        if lock is None:
            return state.get(frame_key)
        with lock:
            return state.get(frame_key)
    return _provider


def start_stream_server(
    state,
    title,
    header,
    footer,
    host,
    port,
    fps,
    jpeg_quality,
    img_width=800,
    sharpen=True,
):
    if "lock" not in state:
        state["lock"] = threading.Lock()
    server = StreamServer(
        build_stream_provider(state),
        host=host,
        port=port,
        fps=fps,
        jpeg_quality=jpeg_quality,
        title=title,
        header=header,
        footer=footer,
        img_width=img_width,
        sharpen=sharpen,
    )
    server.start()
    url = format_stream_url(host, port)
    return server, url
