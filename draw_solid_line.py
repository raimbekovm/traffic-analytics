#!/usr/bin/env python3
"""
Interactive tool to draw multiple solid lines (no-crossing zones).

Controls:
    Left click - Add point
    Enter - Save current line and start next
    'r' - Reset current line
    'q' - Finish and save all
"""

import cv2
import yaml
import numpy as np
from src.video_source import VideoSource


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    video_config = config.get('video', {})

    # Open video
    print(f"Opening video source...")
    video = VideoSource(
        source=video_config['source'],
        source_type=video_config.get('source_type', 'hls')
    )
    video.open()
    success, frame_info = video.read()
    video.release()

    if not success:
        print("Failed to get frame")
        return 1

    frame = frame_info.frame
    temp_frame = frame.copy()
    current_points = []
    all_lines = []

    # Load existing lines
    existing_lines = config.get('violations', {}).get('solid_lines', [])
    line_num = len(existing_lines)

    def redraw():
        nonlocal temp_frame
        temp_frame = frame.copy()

        # Draw existing solid lines (from config)
        for line in existing_lines:
            pts = np.array(line['points'], dtype=np.int32)
            cv2.polylines(temp_frame, [pts], isClosed=False, color=(0, 0, 255), thickness=3)
            if line['points']:
                cv2.putText(temp_frame, line['name'], tuple(line['points'][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw newly added lines (this session)
        for i, line in enumerate(all_lines):
            pts = np.array(line['points'], dtype=np.int32)
            cv2.polylines(temp_frame, [pts], isClosed=False, color=(0, 0, 255), thickness=3)
            if line['points']:
                cv2.putText(temp_frame, line['name'], tuple(line['points'][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw counting lines
        for line in config.get('counting', {}).get('lines', []):
            cv2.line(temp_frame, tuple(line['start']), tuple(line['end']),
                     tuple(line.get('color', [0, 255, 255])), 2)

        # Draw current points
        for i, pt in enumerate(current_points):
            cv2.circle(temp_frame, tuple(pt), 6, (0, 255, 0), -1)
            if i > 0:
                cv2.line(temp_frame, tuple(current_points[i-1]), tuple(pt), (0, 255, 0), 3)

        # Instructions
        cv2.putText(temp_frame, "Click points | Enter=save line | r=reset | q=finish",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(temp_frame, f"Lines added: {len(all_lines)} | Current points: {len(current_points)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append([x, y])
            redraw()
            cv2.imshow('Draw Solid Lines', temp_frame)

    cv2.namedWindow('Draw Solid Lines')
    cv2.setMouseCallback('Draw Solid Lines', mouse_callback)

    print("\n" + "="*50)
    print("DRAW SOLID LINES (No Crossing Zones)")
    print("="*50)
    print("Click points to draw each line")
    print("Enter = Save line and start next")
    print("r = Reset current line")
    print("q = Finish and save all to config")
    print("="*50 + "\n")

    redraw()
    cv2.imshow('Draw Solid Lines', temp_frame)

    while True:
        key = cv2.waitKey(100) & 0xFF

        if key == ord('r') or key == ord('R'):
            current_points = []
            redraw()
            cv2.imshow('Draw Solid Lines', temp_frame)
            print("Reset current line")

        elif key == 13 or key == 10:  # Enter - save line
            if len(current_points) >= 2:
                line_num += 1
                new_line = {
                    'name': f"Solid Line {line_num}",
                    'points': current_points.copy(),
                    'color': [0, 0, 255]
                }
                all_lines.append(new_line)
                print(f"Saved: {new_line['name']} ({len(current_points)} points)")
                current_points = []
                redraw()
                cv2.imshow('Draw Solid Lines', temp_frame)
            else:
                print("Need at least 2 points!")

        elif key == ord('q') or key == ord('Q') or key == 27:  # q or Esc
            break

        # Check if window was closed
        if cv2.getWindowProperty('Draw Solid Lines', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    if not all_lines:
        print("No lines added")
        return 0

    # Save to config
    if 'violations' not in config:
        config['violations'] = {'enabled': True, 'solid_lines': []}
    if 'solid_lines' not in config['violations']:
        config['violations']['solid_lines'] = []

    config['violations']['solid_lines'].extend(all_lines)

    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n{len(all_lines)} solid lines saved to config.yaml")
    return 0


if __name__ == "__main__":
    exit(main())
