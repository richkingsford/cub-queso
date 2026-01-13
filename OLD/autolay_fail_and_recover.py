import argparse
import sys

import autolay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    autolay.STREAM_ENABLED = not args.no_stream
    autolay.DEBUG_MODE = args.debug

    autolay.summarize_all_demos()

    sessions = autolay.find_sessions()
    session = args.session or (sessions[0] if sessions else None)
    if not session:
        print("No sessions found.")
        sys.exit(1)

    try:
        autolay.main_autoplay(session, scenarios=[("FAIL", 5), ("RECOVER", 60)])
    except KeyboardInterrupt:
        if autolay.app_state.robot:
            autolay.app_state.robot.stop()
        autolay.app_state.running = False


if __name__ == "__main__":
    main()
