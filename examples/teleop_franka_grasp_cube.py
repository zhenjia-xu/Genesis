import argparse
import copy
import random
import time

import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
from teleop_utils import IPhoneClient, KeyboardClient

import genesis as gs


def build_scene(vis):
    ########################## init ##########################
    gs.init(
        seed=0,
        precision="32",
        logging_level="debug",
    )
    np.set_printoptions(precision=7, suppress=True)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1),
            camera_lookat=(0.0, -0.5, 0.3),
            camera_fov=50,
            max_FPS=60,
        ),
        show_viewer=vis,
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0, 0, -9.8),
        ),
        show_FPS=False,
    )

    ########################## entities ##########################
    entities = dict()
    entities["plane"] = scene.add_entity(
        gs.morphs.Plane(),
    )
    entities["robot"] = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    entities["cube"] = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.Box(
            pos=(0.2, -0.5, 0.07),
            size=(0.05, 0.05, 0.05),
        ),
        surface=gs.surfaces.Default(color=(0.5, 1, 0.5)),
    )

    entities["target"] = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    ########################## build ##########################
    scene.build()

    return scene, entities


def run_sim(scene, entities, clients, device):
    robot = entities["robot"]
    target_entity = entities["target"]

    robot_init_pos = np.array([0, -0.5, 0.3])
    robot_init_R = R.from_euler("yz", [np.pi, np.pi / 2])
    target_pos = robot_init_pos.copy()
    target_R = robot_init_R

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    ee_link = robot.get_link("hand")

    T_inv_init = np.eye(4)  # for iphone

    def reset_scene():
        nonlocal target_pos, target_R
        target_pos = robot_init_pos.copy()
        target_R = robot_init_R
        target_quat = target_R.as_quat(scalar_first=True)
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.set_qpos(q[:-2], motors_dof)

        entities["cube"].set_pos((random.uniform(-0.2, 0.2), random.uniform(-0.6, -0.4), 0.05))
        entities["cube"].set_quat(R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True))

        if device == "iphone":
            nonlocal T_inv_init
            client: IPhoneClient = clients["iphone"]
            T_init = np.array(client.get_latest_data()["transformMatrix"])
            T_inv_init = np.linalg.inv(T_init)

    if device == "keyboard":
        print("\nKeyboard Controls:")
        print("↑\t- Move Forward (North)")
        print("↓\t- Move Backward (South)")
        print("←\t- Move Left (West)")
        print("→\t- Move Right (East)")
        print("n\t- Move Up")
        print("m\t- Move Down")
        print("j\t- Rotate Counterclockwise")
        print("k\t- Rotate Clockwise")
        print("u\t- Reset Scene")
        print("space\t- Press to close gripper, release to open gripper")
        print("esc\t- Quit")
    elif device == "iphone":
        print("\niPhone Controls:")
        print("Make sure your iPhone and computer are on the same network.")
        print("Note that haptic feedback will be enabled when the close button is pressed.")
        print("Edit Button\t- Enter the IP address of your computer.")
        print("\u21BB Button\t- (Re)start socket connection.")
        print("Reset Button\t- Reset the scene.")
        print("Close Button\t- Press to close the gripper, release to open the gripper.")
        print("------------------------------------------------------------")

        print("Now, please press the \u21BB button to start.")
        client: IPhoneClient = clients["iphone"]
        while len(client.get_latest_data()) == 0:
            time.sleep(0.01)

    # reset scen before starting teleoperation
    reset_scene()

    # start teleoperation
    stop = False
    while not stop:
        pressed_keys = clients["keyboard"].pressed_keys.copy()

        # reset scene:
        reset_flag = False
        reset_flag |= keyboard.KeyCode.from_char("u") in pressed_keys
        if device == "iphone":
            reset_flag |= clients["iphone"].get_latest_data()["buttonStates"].get("Reset", False)
        if reset_flag:
            reset_scene()

        # stop teleoperation
        stop = keyboard.Key.esc in pressed_keys

        # get ee target pose
        is_close_gripper = False
        if device == "keyboard":
            dpos = 0.002
            drot = 0.01
            for key in pressed_keys:
                if key == keyboard.Key.up:
                    target_pos[1] += dpos
                elif key == keyboard.Key.down:
                    target_pos[1] -= dpos
                elif key == keyboard.Key.right:
                    target_pos[0] += dpos
                elif key == keyboard.Key.left:
                    target_pos[0] -= dpos
                elif key == keyboard.KeyCode.from_char("n"):
                    target_pos[2] += dpos
                elif key == keyboard.KeyCode.from_char("m"):
                    target_pos[2] -= dpos
                elif key == keyboard.KeyCode.from_char("j"):
                    target_R = R.from_euler("z", drot) * target_R
                elif key == keyboard.KeyCode.from_char("k"):
                    target_R = R.from_euler("z", -drot) * target_R
                elif key == keyboard.Key.space:
                    is_close_gripper = True
        elif device == "iphone":
            client: IPhoneClient = clients["iphone"]
            latest_data = copy.deepcopy(client.get_latest_data())
            T_delta = T_inv_init @ np.array(latest_data["transformMatrix"])
            target_pos = robot_init_pos + T_delta[:3, 3]
            target_R = R.from_matrix(T_delta[:3, :3]) * robot_init_R

            is_close_gripper = latest_data["buttonStates"].get("Close", False)

        # control arm
        target_quat = target_R.as_quat(scalar_first=True)
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.control_dofs_position(q[:-2], motors_dof)

        # control gripper
        if is_close_gripper:
            robot.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
        else:
            robot.control_dofs_force(np.array([0.5, 0.5]), fingers_dof)

        if device == "iphone":
            client: IPhoneClient = clients["iphone"]
            client.set_haptic_feedback(is_close_gripper)

        scene.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-d", "--device", type=str, default="iphone", choices=["keyboard", "iphone"])
    args = parser.parse_args()

    scene, entities = build_scene(args.vis)

    clients = dict()
    clients["keyboard"] = KeyboardClient()
    if args.device == "iphone":
        clients["iphone"] = IPhoneClient()
        clients["iphone"].start()

    ########################## socket ##########################
    if gs.utils.get_platform() == "macOS":
        gs.tools.run_in_another_thread(fn=run_sim, args=(scene, entities, clients, args.device))
        scene.viewer.start()
    else:
        run_sim(scene, entities, clients, args.device)


if __name__ == "__main__":
    main()
