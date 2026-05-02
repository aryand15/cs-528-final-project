#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys
import time
from collections import deque

import numpy as np
import serial
import serial.tools.list_ports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BAUD_RATE     = 115200
SAMPLE_HZ     = 100
WINDOW_SEC    = 1.0
WINDOW_LEN    = int(SAMPLE_HZ * WINDOW_SEC)
GESTURES_DIR  = "gestures"

LABELS = [
  "hold_item",
  "item_forwards",
  "item_backwards",
  "look_backwards",
  "drift_hop",
]

ACTIVITY_THRESHOLD = 0.15
ACTIVITY_WINDOW    = 20
COOLDOWN_SEC       = 1.0

LINE_RE = re.compile(
  r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
  r"\s*\|\s*"
  r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
)


def parse_line(line: str):
  m = LINE_RE.search(line)
  if m:
    return tuple(float(m.group(k)) for k in ("ax", "ay", "az", "gx", "gy", "gz"))
  return None


def find_port() -> str:
  ports = serial.tools.list_ports.comports()
  usb = [p for p in ports if "usb" in p.device.lower() or "usbserial" in p.device.lower()]
  if usb:
    return usb[0].device
  if ports:
    return ports[0].device
  print("[ERROR] No serial ports found. Plug in your ESP32 or specify --port.", file=sys.stderr)
  sys.exit(1)


def features(window) -> np.ndarray:
  a = np.asarray(window, dtype=np.float32)
  return np.concatenate([
    a.mean(axis=0),
    a.std(axis=0),
    a.min(axis=0),
    a.max(axis=0),
  ])


def load_dataset(gestures_dir: str, labels):
  X, y = [], []
  for label in labels:
    files = sorted(glob.glob(os.path.join(gestures_dir, f"{label}_*.txt")))
    if not files:
      print(f"[WARN] No files found for label '{label}' in {gestures_dir}/")
      continue
    kept = 0
    for path in files:
      try:
        arr = np.loadtxt(path, delimiter=",")
      except Exception as e:
        print(f"[WARN] Skipping {path}: {e}")
        continue
      if arr.ndim != 2 or arr.shape[1] != 6 or arr.shape[0] < 10:
        continue
      X.append(features(arr))
      y.append(label)
      kept += 1
    print(f"[LOAD] {label:16s} {kept:3d} files")
  return np.array(X), np.array(y)


def train_model(X, y):
  model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale")),
  ])
  model.fit(X, y)
  print(f"[TRAIN] Training accuracy: {model.score(X, y):.3f}")
  return model


def read_sample(ser):
  raw = ser.readline()
  try:
    line = raw.decode("utf-8", errors="replace").strip()
  except Exception:
    return None
  return parse_line(line)


def realtime_loop(model, port: str, baud: int):
  print(f"[INFO] Opening {port} @ {baud} baud …")
  with serial.Serial(port, baud, timeout=1) as ser:
    print("[INFO] Listening for gestures (Ctrl-C to stop)")
    recent = deque(maxlen=ACTIVITY_WINDOW)

    while True:
      sample = read_sample(ser)
      if sample is None:
        continue
      recent.append(sample)

      if len(recent) < ACTIVITY_WINDOW:
        continue

      arr = np.asarray(recent, dtype=np.float32)
      activity = np.linalg.norm(arr[:, :3], axis=1).std()
      if activity < ACTIVITY_THRESHOLD:
        continue

      window = list(recent)
      while len(window) < WINDOW_LEN:
        s = read_sample(ser)
        if s is not None:
          window.append(s)

      feats = features(window[:WINDOW_LEN]).reshape(1, -1)
      pred = model.predict(feats)[0]
      print(f"[GESTURE] {pred}   (activity={activity:.3f})")

      t0 = time.perf_counter()
      while time.perf_counter() - t0 < COOLDOWN_SEC:
        read_sample(ser)
      recent.clear()


def main():
  parser = argparse.ArgumentParser(description="Real-time IMU gesture recognition")
  parser.add_argument("--baud", default=BAUD_RATE, type=int)
  parser.add_argument("--gestures-dir", default=GESTURES_DIR,
                      help=f"Directory of recorded gesture .txt files (default: {GESTURES_DIR}/)")
  args = parser.parse_args()

  print("[INFO] Loading training data …")
  X, y = load_dataset(args.gestures_dir, LABELS)
  if len(X) == 0:
    print("[ERROR] No training samples loaded.", file=sys.stderr)
    sys.exit(1)
  print(f"[INFO] Loaded {len(X)} samples across {len(set(y))} classes")

  print("[INFO] Training SVM …")
  model = train_model(X, y)

  port = find_port()
  try:
    realtime_loop(model, port, args.baud)
  except KeyboardInterrupt:
    print("\n[INFO] Bye.")


if __name__ == "__main__":
  main()
