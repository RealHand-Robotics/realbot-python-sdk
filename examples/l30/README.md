# L30 examples

Run read-only examples directly from the SDK root, for example:

```bash
python3 examples/l30/read_info.py
python3 examples/l30/read_touch.py
python3 examples/l30/stream.py
```

They default to `--side left --canfd-id 0`. Override them with flags or the
`L30_SIDE` and `CANFD_ID` environment variables.

Command-changing examples run directly. For example:

```bash
python3 examples/l30/set_speed.py --value 75
python3 examples/l30/set_torque_limit.py --percent 50
python3 examples/l30/send_position.py --positions 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

`send_position.py` validates the connected hand's native per-joint limits and
disables the joints in a `finally` block. `emergency_stop.py` sends a real
emergency-stop command and is only for an actual emergency.
