#!/usr/bin/env python3
"""Autobuild orchestration using telemetry-backed logic."""
import argparse
import threading

from helper_streaming import start_stream_server
from telemetry_process import *  # noqa: F401,F403
from telemetry_process import (
    run_alignment_segment,
    replay_segment,
    select_demo_segment,
    collect_segments,
    update_process_model_from_demos,
    refresh_autobuild_config,
    load_process_model,
    format_gate_lines,
    format_headline,
    normalize_step_label,
    StepState,
    WorldModel,
    Robot,
    ArucoBrickVision,
    DEMO_DIR,
    PROCESS_MODEL_FILE,
    MAX_PHASE_ATTEMPTS,
    load_demo_logs,
    STREAM_HOST,
    STREAM_PORT,
    STREAM_FPS,
    STREAM_JPEG_QUALITY,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_WHITE,
    USE_LEARNED_POLICY,
)


def run_autobuild(session_name=None, stream=True):
    logs = load_demo_logs(DEMO_DIR, session_name)
    update_process_model_from_demos(logs, PROCESS_MODEL_FILE)
    refresh_autobuild_config(PROCESS_MODEL_FILE)
    if not logs:
        print("[AUTO] No demo logs found after update.")
        return

    segments_by_obj, _ = collect_segments(logs)
    model = load_process_model(PROCESS_MODEL_FILE)
    steps = list((model.get("steps") or {}).keys())
    if not steps:
        print("[AUTO] No steps defined in process model.")
        return

    robot = Robot()
    vision = ArucoBrickVision(debug=False)
    world = WorldModel()
    world.suppress_brick_state_log = True
    world.log_brick_frames = False
    world.log_fresh_frames = True

    stream_server = None
    if stream:
        stream_state = {"frame": None, "lock": threading.Lock()}
        world._stream_state = stream_state
        stream_server, url = start_stream_server(
            stream_state,
            title="Robot Leia - Autobuild",
            header="Robot Leia - Autobuild",
            footer="Use the terminal for logs. Keep this window open to see the live feed.",
            host=STREAM_HOST,
            port=STREAM_PORT,
            fps=STREAM_FPS,
            jpeg_quality=STREAM_JPEG_QUALITY,
        )
        print(format_headline(f"[VISION] Stream started at {url}", COLOR_GREEN))
    else:
        print(format_headline("[VISION] Stream disabled", COLOR_WHITE))

    try:
        for obj_name in steps:
            normalized = normalize_step_label(obj_name)
            if normalized in StepState.__members__:
                world.step_state = StepState[normalized]
            else:
                world.step_state = StepState.FIND_BRICK

            cfg = world.process_rules.get(normalized, {}) if world.process_rules else {}
            nominal_only = bool(cfg.get("nominalDemosOnly"))
            segment, seg_type = select_demo_segment(segments_by_obj, normalized, nominal_only)
            if not segment:
                print(format_headline(f"[FAIL] {normalized}: no demo segment found", COLOR_RED))
                return

            attempts = 0
            last_reason = None
            while attempts < MAX_PHASE_ATTEMPTS:
                header = (
                    f"Attempting {normalized} "
                    f"(attempt {attempts + 1}/{MAX_PHASE_ATTEMPTS}; demo {seg_type})"
                )
                print(format_headline(header, COLOR_GREEN))
                start_desc, success_desc = format_gate_lines(cfg)
                print(f"  start gates: {start_desc}")
                print(f"  success gates: {success_desc}")
                ok, reason = replay_segment(segment, normalized, robot, vision, world)
                if ok:
                    break
                attempts += 1
                last_reason = reason or "unknown"
                print(format_headline(f"[RETRY] {normalized} ({last_reason})", COLOR_RED))
            if attempts >= MAX_PHASE_ATTEMPTS:
                reason_suffix = f" ({last_reason})" if last_reason else ""
                print(
                    format_headline(
                        f"[FAIL] {normalized} failed after {attempts} attempts{reason_suffix}",
                        COLOR_RED,
                    )
                )
                return

        print("[JOB] SUCCESS")
    finally:
        robot.stop()
        vision.close()
        if stream_server:
            stream_server.stop()


def main():
    parser = argparse.ArgumentParser(description="Robot Leia Autobuild")
    parser.add_argument("--session", help="demo session file or folder", default=None)
    parser.add_argument("--learn", help="enable learning from demonstration policy", action="store_true")
    parser.add_argument("--stream", dest="stream", action="store_true",
                        help="Enable livestreaming")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="Disable livestreaming")
    parser.set_defaults(stream=True)
    args = parser.parse_args()

    global USE_LEARNED_POLICY
    if args.learn:
        USE_LEARNED_POLICY = True

    run_autobuild(session_name=args.session, stream=args.stream)
    print("\n" * 5, end="")


if __name__ == "__main__":
    main()
