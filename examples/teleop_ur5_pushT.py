import argparse
import copy
import random
import time

import numpy as np
import torch
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
from teleop_utils import IPhoneClient, KeyboardClient

import genesis as gs


def build_scene(vis):
    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")
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
        morph=gs.morphs.MJCF(file="xml/universal_robots_ur5e/ur5e.xml"),
    )
    entities["end_effector"] = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.Cylinder(pos=(0, 0, 0.1), radius=0.0125, height=0.2, fixed=True, requires_jac_and_IK=True),
    )
    scene.link_entities(entities["robot"], entities["end_effector"], "ee_virtual_link", "cylinder_baselink")

    T_object_material = gs.materials.Rigid(rho=300)
    T_object_surface = gs.surfaces.Default(color=(1, 0.5, 0.5))
    entities["T_object"] = scene.add_entity(
        material=T_object_material,
        morph=gs.morphs.Box(
            pos=(0.5, 0, 0.02),
            size=(0.2, 0.05, 0.04),
        ),
        surface=T_object_surface,
    )
    T_object_part2 = scene.add_entity(
        material=T_object_material,
        morph=gs.morphs.Box(pos=(0.0, -0.1, 0.0), size=(0.05, 0.15, 0.04), fixed=True),
        surface=T_object_surface,
    )
    scene.link_entities(entities["T_object"], T_object_part2, "box_baselink", "box_baselink")

    # add
    T_target_material = gs.materials.Rigid(rho=300)
    T_target_surface = gs.surfaces.Default(color=(0.5, 1, 0.5), opacity=0.5)
    T_target_height = 0.004
    entities["T_target"] = scene.add_entity(
        material=T_target_material,
        morph=gs.morphs.Box(
            pos=(0.5, 0, T_target_height / 2), size=(0.2, 0.05, T_target_height), collision=False, fixed=True
        ),
        surface=T_target_surface,
    )
    T_target_part2 = scene.add_entity(
        material=T_target_material,
        morph=gs.morphs.Box(pos=(0.0, 0.1, 0.0), size=(0.05, 0.15, T_target_height), collision=False, fixed=True),
        surface=T_target_surface,
    )
    scene.link_entities(entities["T_target"], T_target_part2, "box_baselink", "box_baselink")

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

    robot_init_pos = np.array([0, -0.5, 0.205])
    robot_init_R = R.from_euler("y", np.pi)
    target_pos = robot_init_pos.copy()
    target_R = robot_init_R

    end_effector = entities["end_effector"]

    motors_dof = np.arange(6)
    ee_link = robot.get_link("ee_virtual_link")

    T_inv_init = np.eye(4)  # for iphone

    def reset_scene():
        nonlocal target_pos, target_R
        target_pos = robot_init_pos.copy()
        target_R = robot_init_R
        target_quat = target_R.as_quat(scalar_first=True)
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.set_qpos(q, motors_dof)

        entities["T_object"].set_pos((random.uniform(-0.2, 0.2), random.uniform(-0.6, -0.4), 0.02))
        entities["T_object"].set_quat(R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True))

        entities["T_target"].set_pos((random.uniform(-0.2, 0.2), random.uniform(-0.6, -0.4), 0.002))
        entities["T_target"].set_quat(R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True))

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
        print("u\t- Reset Scene")
        print("esc\t- Quit")
    elif device == "iphone":
        print("\niPhone Controls:")
        print("Make sure your iPhone and computer are on the same network.")
        print("Note that haptic feedback will be enabled when the end-effector has contact with the object.")
        print("Edit Button\t- Enter the IP address of your computer.")
        print("\u21BB Button\t- (Re)start socket connection.")
        print("Reset Button\t- Reset the scene.")
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
        if device == "keyboard":
            dpos = 0.002
            for key in pressed_keys:
                if key == keyboard.Key.up:
                    target_pos[1] += dpos
                elif key == keyboard.Key.down:
                    target_pos[1] -= dpos
                elif key == keyboard.Key.right:
                    target_pos[0] += dpos
                elif key == keyboard.Key.left:
                    target_pos[0] -= dpos
        elif device == "iphone":
            client: IPhoneClient = clients["iphone"]
            latest_data = copy.deepcopy(client.get_latest_data())
            T_delta = T_inv_init @ np.array(latest_data["transformMatrix"])

            dpos = T_delta[:3, 3]
            dpos[2] = 0
            target_pos = robot_init_pos + dpos

        # control arm
        target_quat = target_R.as_quat(scalar_first=True)
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.control_dofs_position(q, motors_dof)

        if device == "iphone":
            client: IPhoneClient = clients["iphone"]
            is_contact = torch.norm(end_effector.get_links_net_contact_force()) > 1e-8
            client.set_haptic_feedback(is_contact)

        scene.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-d", "--device", type=str, default="keyboard", choices=["keyboard", "iphone"])
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
