import os

from realhand import P7

TCP_HOST = os.environ.get("P7_TCP_HOST", "192.168.10.21")
SIDE = os.environ.get("P7_SIDE", "left")


with P7(side=SIDE, tcp_host=TCP_HOST) as arm:
    state = arm.get_state()

    print(f"P7 {SIDE} SDK state")
    print("Pose:", state.pose)
    print("Joint angles:", [joint.angle for joint in state.joint_angles])
    print("Joint velocities:", [velocity.velocity for velocity in state.joint_velocities])
    print("Joint torques:", [torque.torque for torque in state.joint_torques])
    print(
        "Joint temperatures:",
        [temperature.temperature for temperature in state.joint_temperatures],
    )

    raw_state = arm.robot.get_state()
    if raw_state is not None:
        print("\nRaw RBot full state:")
        print(raw_state.to_dict())
