#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys
import threading
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

LABELS_1 = [
  "hold_item",
  "item_forwards",
  "item_backwards",
  "look_backwards",
  "drift_hop",
]

LABELS_2 = [
  "accelerate",
  "brake",
]

ACTIVITY_THRESHOLD_HAND = 0.15
ACTIVITY_THRESHOLD_FOOT = 0.05
DRIFT_HOP_ACTIVITY_THRESHOLD = 0.18
ACTIVITY_WINDOW         = 20
COOLDOWN_SEC            = 1.0
BUTTON_TAP_SEC          = 1.5

CONTROLLER_NAMES = {
  1: "left hand",
  2: "right hand",
  3: "foot",
}

DETECT_SEC          = 2.0
DETECT_MIN_ACTIVITY = 0.03

STEER_CALIB_SEC   = 2.0
STEER_ALPHA       = 0.98
STEER_PRINT_HZ    = 60.0
STEER_BAR_WIDTH   = 41
STEER_BAR_MAX_DEG = 90.0

LINE_RE = re.compile(
  r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
  r"\s*\|\s*"
  r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
)

PRINT_LOCK = threading.Lock()


def log(message: str = "", end: str = "\n"):
  with PRINT_LOCK:
    print(message, end=end, flush=True)


def parse_line(line: str):
  m = LINE_RE.search(line)
  if m:
    try:
      return tuple(float(m.group(k)) for k in ("ax", "ay", "az", "gx", "gy", "gz"))
    except ValueError:
      return None
  return None


def read_sample(ser):
  raw = ser.readline()
  try:
    line = raw.decode("utf-8", errors="replace").strip()
  except Exception:
    return None
  return parse_line(line)


def serial_port_candidates():
  ports = sorted(serial.tools.list_ports.comports(), key=lambda p: p.device)
  usb = [
    p for p in ports
    if "usb" in f"{p.device} {p.description} {p.hwid}".lower()
  ]
  return usb or ports


def describe_port(port_info) -> str:
  description = getattr(port_info, "description", "") or "serial device"
  return f"{port_info.device} ({description})"


def open_setup_serials(baud: int):
  serials = {}
  for port_info in serial_port_candidates():
    try:
      ser = serial.Serial(port_info.device, baud, timeout=0.01)
      ser.reset_input_buffer()
      serials[port_info.device] = ser
      log(f"[SETUP] Found {describe_port(port_info)}")
    except serial.SerialException as e:
      log(f"[WARN] Could not open {port_info.device}: {e}")
  if len(serials) < 3:
    log("[ERROR] Need at least 3 open serial ports for the three controllers.")
    sys.exit(1)
  return serials


def score_port_activity(samples) -> float:
  if len(samples) < 5:
    return 0.0
  arr = np.asarray(samples, dtype=np.float32)
  accel_activity = float(np.linalg.norm(arr[:, :3], axis=1).std())
  gyro_activity = float(np.linalg.norm(arr[:, 3:], axis=1).std())
  return accel_activity + 0.01 * gyro_activity


def detect_moving_port(serials, duration_sec: float):
  for ser in serials.values():
    ser.reset_input_buffer()

  samples_by_port = {port: [] for port in serials}
  end_at = time.perf_counter() + duration_sec
  while time.perf_counter() < end_at:
    got_sample = False
    for port, ser in serials.items():
      sample = read_sample(ser)
      if sample is not None:
        samples_by_port[port].append(sample)
        got_sample = True
    if not got_sample:
      time.sleep(0.005)

  scores = {
    port: score_port_activity(samples)
    for port, samples in samples_by_port.items()
  }
  if not scores:
    return None, 0.0, scores
  detected_port, activity = max(scores.items(), key=lambda item: item[1])
  return detected_port, activity, scores


def assign_controller_ports(baud: int):
  log("[SETUP] Plug in all 3 USB controllers now.")
  input("[SETUP] Press Enter when all three are connected: ")

  serials = open_setup_serials(baud)
  assigned = {}
  remaining = dict(serials)
  try:
    while len(assigned) < 3:
      log("")
      log("[SETUP] Move exactly one unassigned controller.")
      input(f"[SETUP] Press Enter, then keep it moving for {DETECT_SEC:.1f}s: ")
      detected_port, activity, scores = detect_moving_port(remaining, DETECT_SEC)

      log("[SETUP] Activity by port:")
      for port, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        marker = "<-- detected" if port == detected_port else ""
        log(f"  {port:16s} activity={score:.3f} {marker}")

      if detected_port is None or activity < DETECT_MIN_ACTIVITY:
        log("[WARN] I did not see enough motion. Try that controller again.")
        continue

      choice = ""
      while choice not in ("1", "2", "3", "r"):
        choice = input(
          f"[SETUP] Detected {detected_port}. Assign it to "
          "1=left hand, 2=right hand, 3=foot, or r=retry: "
        ).strip().lower()

      if choice == "r":
        continue

      controller_id = int(choice)
      if controller_id in assigned:
        log(f"[WARN] Controller {controller_id} is already assigned to {assigned[controller_id]}.")
        continue

      assigned[controller_id] = detected_port
      remaining.pop(detected_port)
      log(f"[SETUP] Controller {controller_id} ({CONTROLLER_NAMES[controller_id]}) -> {detected_port}")

    log("")
    log("[SETUP] Controller mapping complete:")
    for controller_id in sorted(assigned):
      log(f"  {controller_id}) {CONTROLLER_NAMES[controller_id]:10s} -> {assigned[controller_id]}")
    return assigned
  finally:
    for ser in serials.values():
      ser.close()


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
      log(f"[WARN] No files found for label '{label}' in {gestures_dir}/")
      continue
    kept = 0
    for path in files:
      try:
        arr = np.loadtxt(path, delimiter=",")
      except Exception as e:
        log(f"[WARN] Skipping {path}: {e}")
        continue
      if arr.ndim != 2 or arr.shape[1] != 6 or arr.shape[0] < 10:
        continue
      X.append(features(arr))
      y.append(label)
      kept += 1
    log(f"[LOAD] {label:16s} {kept:3d} files")
  return np.array(X), np.array(y)


def train_model(X, y):
  model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale")),
  ])
  model.fit(X, y)
  log(f"[TRAIN] Training accuracy: {model.score(X, y):.3f}")
  return model


def train_required_model(gestures_dir: str, labels, name: str):
  log(f"[INFO] Loading {name} training data ...")
  X, y = load_dataset(gestures_dir, labels)
  if len(X) == 0:
    log(f"[ERROR] No training samples loaded for {name}.")
    sys.exit(1)
  log(f"[INFO] Loaded {len(X)} {name} samples across {len(set(y))} classes")

  log(f"[INFO] Training {name} SVM ...")
  return train_model(X, y)


def should_stop(stop_event) -> bool:
  return stop_event is not None and stop_event.is_set()


def wrap180(deg: float) -> float:
  return ((deg + 180.0) % 360.0) - 180.0


def steer_bar(angle_deg: float) -> str:
  half = STEER_BAR_WIDTH // 2
  pos = int(round((angle_deg / STEER_BAR_MAX_DEG) * half))
  pos = max(-half, min(half, pos))
  cells = ["-"] * STEER_BAR_WIDTH
  cells[half] = "|"
  cells[half + pos] = "#"
  return "[" + "".join(cells) + "]"


class NullOutput:
  def set_steering(self, angle_deg: float):
    pass

  def handle_gesture(self, gesture: str):
    pass

  def release_all(self):
    pass


class XboxGamepadOutput:
  def __init__(self, invert_steering: bool = False):
    try:
      import vgamepad as vg
    except ImportError:
      log("[ERROR] Xbox output needs the 'vgamepad' package and ViGEmBus installed.")
      log("[ERROR] Re-run with --output none to test recognition without controller output.")
      sys.exit(1)

    self.vg = vg
    self.gamepad = vg.VX360Gamepad()
    self.lock = threading.Lock()
    self.invert_steering = invert_steering
    self.steer_x = 0.0
    self.right_trigger = 0.0
    self.left_trigger = 0.0
    self.item_held = False
    self.release_all()
    log("[INFO] Xbox controller output enabled.")

  def _update_locked(self):
    self.gamepad.left_joystick_float(x_value_float=self.steer_x, y_value_float=0.0)
    self.gamepad.right_trigger_float(value_float=self.right_trigger)
    self.gamepad.left_trigger_float(value_float=self.left_trigger)
    self.gamepad.update()

  def _log_input(self, message: str):
    log(f"[XBOX] {message}")

  def set_steering(self, angle_deg: float):
    x = max(-1.0, min(1.0, angle_deg / STEER_BAR_MAX_DEG))
    if self.invert_steering:
      x = -x
    with self.lock:
      self.steer_x = x
      self._update_locked()

  def tap_button(self, button, button_name: str):
    def tap_task():
      with self.lock:
        self._log_input(f"tap {button_name}")
        self.gamepad.press_button(button=button)
        self._update_locked()
      time.sleep(BUTTON_TAP_SEC)
      with self.lock:
        self.gamepad.release_button(button=button)
        self._update_locked()
    threading.Thread(target=tap_task, daemon=True).start()

  def _release_item_locked(self):
    if self.item_held:
      self._log_input("release LB")
      self.gamepad.release_button(button=self.vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
      self.item_held = False

  def handle_gesture(self, gesture: str):
    b = self.vg.XUSB_BUTTON
    if gesture == "hold_item":
      with self.lock:
        self._log_input("hold LB")
        self.gamepad.press_button(button=b.XUSB_GAMEPAD_LEFT_SHOULDER)
        self.item_held = True
        self._update_locked()
    elif gesture == "item_forwards":
      with self.lock:
        was_held = self.item_held
        if was_held:
          self._release_item_locked()
          self._update_locked()
      if not was_held:
        self.tap_button(b.XUSB_GAMEPAD_LEFT_SHOULDER, "LB")
    elif gesture == "item_backwards":
      def backwards_task():
        with self.lock:
          self._log_input("hold D-pad down")
          self.gamepad.press_button(button=b.XUSB_GAMEPAD_DPAD_DOWN)
          was_held = self.item_held
          if was_held:
            self._release_item_locked()
          else:
            self._log_input("tap LB")
            self.gamepad.press_button(button=b.XUSB_GAMEPAD_LEFT_SHOULDER)
          self._update_locked()
        time.sleep(BUTTON_TAP_SEC)
        with self.lock:
          if not was_held:
            self.gamepad.release_button(button=b.XUSB_GAMEPAD_LEFT_SHOULDER)
          self._log_input("release D-pad down")
          self.gamepad.release_button(button=b.XUSB_GAMEPAD_DPAD_DOWN)
          self._update_locked()
      threading.Thread(target=backwards_task, daemon=True).start()
    elif gesture == "look_backwards":
      self.tap_button(b.XUSB_GAMEPAD_Y, "Y")
    elif gesture == "drift_hop":
      self.tap_button(b.XUSB_GAMEPAD_RIGHT_SHOULDER, "RB")
    elif gesture == "accelerate":
      with self.lock:
        self._log_input("hold B, release A")
        self.gamepad.press_button(button=b.XUSB_GAMEPAD_B)
        self.gamepad.release_button(button=b.XUSB_GAMEPAD_A)
        self._update_locked()
    elif gesture == "brake":
      with self.lock:
        self._log_input("hold A, release B")
        self.gamepad.press_button(button=b.XUSB_GAMEPAD_A)
        self.gamepad.release_button(button=b.XUSB_GAMEPAD_B)
        self._update_locked()

  def release_all(self):
    with self.lock:
      self.steer_x = 0.0
      self.right_trigger = 0.0
      self.left_trigger = 0.0
      self.item_held = False
      try:
        self.gamepad.release_button(button=self.vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.release_button(button=self.vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.gamepad.reset()
      except AttributeError:
        pass
      self._update_locked()


def make_output(output_mode: str, invert_steering: bool):
  if output_mode == "none":
    log("[INFO] Controller output disabled.")
    return NullOutput()
  if output_mode == "xbox":
    return XboxGamepadOutput(invert_steering=invert_steering)
  raise ValueError(f"Unsupported output mode: {output_mode}")


def steering_loop(port: str, baud: int, output=None, stop_event=None, name: str = "steering"):
  log(f"[{name}] Opening {port} @ {baud} baud ...")
  with serial.Serial(port, baud, timeout=0.2) as ser:
    log(f"[{name}] Hold the remote level for {STEER_CALIB_SEC:.1f}s to calibrate ...")
    calib = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < STEER_CALIB_SEC and not should_stop(stop_event):
      s = read_sample(ser)
      if s is not None:
        calib.append(s)
    if should_stop(stop_event):
      return
    if len(calib) < 10:
      log(f"[{name}] ERROR: Not enough calibration samples; is the IMU streaming?")
      return

    arr = np.asarray(calib, dtype=np.float32)
    ax0, ay0, _ = arr[:, 0:3].mean(axis=0)
    gz_bias = float(arr[:, 5].mean())
    zero_angle = float(np.degrees(np.arctan2(ax0, ay0)))
    log(f"[{name}] Calibrated zero={zero_angle:+.2f} deg  gz_bias={gz_bias:+.3f} deg/s  "
        f"({len(calib)} samples)")
    log(f"[{name}] Steering active - rotate the remote around its z-axis (Ctrl-C to stop)")

    angle = 0.0
    last_t = time.perf_counter()
    last_print = 0.0
    print_period = 1.0 / STEER_PRINT_HZ

    while not should_stop(stop_event):
      # Drain the input buffer completely to ensure we only process the freshest sample
      while ser.in_waiting > 100:
        ser.reset_input_buffer()
        ser.readline() # Discard the partial line
      
      sample = read_sample(ser)
      if sample is None:
        continue
      ax, ay, _, _, _, gz = sample
      now = time.perf_counter()
      dt = now - last_t
      last_t = now
      if dt <= 0 or dt > 0.5:
        continue

      accel_angle = wrap180(np.degrees(np.arctan2(ax, ay)) - zero_angle)
      gyro_rate   = gz - gz_bias

      predicted = angle + gyro_rate * dt
      err       = wrap180(accel_angle - predicted)
      angle     = wrap180(predicted + (1.0 - STEER_ALPHA) * err)

      if now - last_print >= print_period:
        last_print = now
        if output is not None:
          output.set_steering(angle)
        log(f"\r[{name}] {angle:+7.2f} deg  rate={gyro_rate:+7.2f} deg/s  {steer_bar(angle)}",
            end="")


def realtime_loop(model, port: str, baud: int, activity_threshold: float,
                  label_activity_thresholds=None, output=None, stop_event=None,
                  name: str = "gesture"):
  log(f"[{name}] Opening {port} @ {baud} baud ...")
  with serial.Serial(port, baud, timeout=0.2) as ser:
    log(f"[{name}] Listening for gestures (Ctrl-C to stop)")
    recent = deque(maxlen=ACTIVITY_WINDOW)

    while not should_stop(stop_event):
      if ser.in_waiting > 200:
        ser.reset_input_buffer()
        ser.readline() # Discard the partial line
      
      sample = read_sample(ser)
      if sample is None:
        continue
      recent.append(sample)

      if len(recent) < ACTIVITY_WINDOW:
        continue

      arr = np.asarray(recent, dtype=np.float32)
      activity = np.linalg.norm(arr[:, :3], axis=1).std()
      if activity < activity_threshold:
        continue

      window = list(recent)
      while len(window) < WINDOW_LEN and not should_stop(stop_event):
        s = read_sample(ser)
        if s is not None:
          window.append(s)
      if len(window) < WINDOW_LEN:
        break

      feats = features(window[:WINDOW_LEN]).reshape(1, -1)
      pred = model.predict(feats)[0]
      label_threshold = (label_activity_thresholds or {}).get(pred)
      if label_threshold is not None and activity < label_threshold:
        recent.clear()
        continue
      if output is not None:
        output.handle_gesture(pred)
      log(f"[{name}] GESTURE {pred}   (activity={activity:.3f})")

      t0 = time.perf_counter()
      while time.perf_counter() - t0 < COOLDOWN_SEC and not should_stop(stop_event):
        read_sample(ser)
      recent.clear()


def controller_worker(name: str, stop_event, target, *args):
  try:
    target(*args, stop_event=stop_event, name=name)
  except serial.SerialException as e:
    log(f"\n[{name}] ERROR: Serial connection failed: {e}")
    stop_event.set()
  except Exception as e:
    log(f"\n[{name}] ERROR: {e}")
    stop_event.set()


def run_all_controllers(assignments, right_hand_model, foot_model, baud: int, output):
  stop_event = threading.Event()
  workers = [
    threading.Thread(
      target=controller_worker,
      args=("left hand", stop_event, steering_loop, assignments[1], baud, output),
      daemon=False,
    ),
    threading.Thread(
      target=controller_worker,
      args=("right hand", stop_event, realtime_loop,
            right_hand_model, assignments[2], baud, ACTIVITY_THRESHOLD_HAND,
            {"drift_hop": DRIFT_HOP_ACTIVITY_THRESHOLD}, output),
      daemon=False,
    ),
    threading.Thread(
      target=controller_worker,
      args=("foot", stop_event, realtime_loop,
            foot_model, assignments[3], baud, ACTIVITY_THRESHOLD_FOOT, None, output),
      daemon=False,
    ),
  ]

  log("[INFO] Starting all 3 controllers.")
  for worker in workers:
    worker.start()

  try:
    while any(worker.is_alive() for worker in workers):
      time.sleep(0.2)
      if stop_event.is_set():
        break
  except KeyboardInterrupt:
    log("\n[INFO] Stopping all controllers ...")
    stop_event.set()

  for worker in workers:
    worker.join(timeout=2.0)
  output.release_all()
  log("\n[INFO] Bye.")


def main():
  parser = argparse.ArgumentParser(description="Real-time IMU gesture recognition")
  parser.add_argument("--baud", default=BAUD_RATE, type=int)
  parser.add_argument("--gestures-dir", default=GESTURES_DIR,
                      help=f"Directory of recorded gesture .txt files (default: {GESTURES_DIR}/)")
  parser.add_argument("--output", choices=("xbox", "none"), default="xbox",
                      help="Send recognized controls to a virtual Xbox controller, or disable output.")
  parser.add_argument("--invert-steering", action="store_true",
                      help="Flip the left-hand steering direction.")
  args = parser.parse_args()

  output = make_output(args.output, args.invert_steering)
  assignments = assign_controller_ports(args.baud)

  right_hand_model = train_required_model(args.gestures_dir, LABELS_1, "right hand")
  foot_model = train_required_model(args.gestures_dir, LABELS_2, "foot")

  run_all_controllers(assignments, right_hand_model, foot_model, args.baud, output)


if __name__ == "__main__":
  main()
