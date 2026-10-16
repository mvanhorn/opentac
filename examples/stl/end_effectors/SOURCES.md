# End-effector mesh sources

Converted to binary STL in millimetres via trimesh (metres-scale
sources multiplied by 1000).

Each mesh keeps the license of the repository it came from, not OpenTac's.
The license column was checked on 2026-10-16 against the repository's LICENSE
file, or where there is none, its ROS package manifest or README. Three
sources have no LICENSE file:

- `allegro/`: the ROS manifest of `allegro_hand_description` says BSD.
- `inspire_hand/`: the README says the Inspire Hand assets follow dex-urdf's
  table, which gives CC BY-NC-SA 4.0 (non-commercial, share-alike).
- `yale_openhand/`: `openhand_simulation` states no license. Yale's OpenHand
  hardware repository, with the same hand designs, is CC BY-NC 3.0
  (non-commercial).

| robot | part | source (repo@branch:path) | units | license |
|---|---|---|---|---|
| so101 | moving_jaw | TheRobotStudio/SO-ARM100@main:STL/SO101/Individual/Moving_Jaw_SO101.stl | mm | [Apache-2.0](https://github.com/TheRobotStudio/SO-ARM100/blob/main/LICENSE) |
| so101 | fixed_jaw_wrist_roll | TheRobotStudio/SO-ARM100@main:STL/SO101/Individual/Wrist_Roll_Follower_SO101.stl | mm | [Apache-2.0](https://github.com/TheRobotStudio/SO-ARM100/blob/main/LICENSE) |
| robotiq_2f85 | pad | ros-industrial/robotiq@kinetic-devel:robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_pad.dae | mm | [BSD-2-Clause](https://github.com/ros-industrial/robotiq/blob/kinetic-devel/LICENSE) |
| robotiq_2f85 | inner_finger | ros-industrial/robotiq@kinetic-devel:robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_inner_finger.dae | mm | [BSD-2-Clause](https://github.com/ros-industrial/robotiq/blob/kinetic-devel/LICENSE) |
| robotiq_2f140 | inner_finger | ros-industrial/robotiq@kinetic-devel:robotiq_2f_140_gripper_visualization/meshes/visual/robotiq_arg2f_140_inner_finger.stl | m->mm | [BSD-2-Clause](https://github.com/ros-industrial/robotiq/blob/kinetic-devel/LICENSE) |
| franka | fingertip | frankaemika/franka_description@main:meshes/robot_ee/franka_hand_white/visual/finger.dae | m->mm | [Apache-2.0](https://github.com/frankaemika/franka_description/blob/main/LICENSE) |
| franka | hand_body | frankaemika/franka_description@main:meshes/robot_ee/franka_hand_white/collision/hand.stl | m->mm | [Apache-2.0](https://github.com/frankaemika/franka_description/blob/main/LICENSE) |
| franka | arm_link4 | frankaemika/franka_description@main:meshes/robots/fer/visual/link4.dae | m->mm | [Apache-2.0](https://github.com/frankaemika/franka_description/blob/main/LICENSE) |
| franka | arm_link5 | frankaemika/franka_description@main:meshes/robots/fer/visual/link5.dae | m->mm | [Apache-2.0](https://github.com/frankaemika/franka_description/blob/main/LICENSE) |
| leap_hand | fingertip | leap-hand/LEAP_Hand_Sim@master:assets/leap_hand/fingertip.stl | m->mm | [MIT](https://github.com/leap-hand/LEAP_Hand_Sim/blob/master/LICENSE.txt) |
| leap_hand | thumb_fingertip | leap-hand/LEAP_Hand_Sim@master:assets/leap_hand/thumb_fingertip.stl | m->mm | [MIT](https://github.com/leap-hand/LEAP_Hand_Sim/blob/master/LICENSE.txt) |
| leap_hand | palm | leap-hand/LEAP_Hand_Sim@master:assets/leap_hand/palm_lower.stl | m->mm | [MIT](https://github.com/leap-hand/LEAP_Hand_Sim/blob/master/LICENSE.txt) |
| leap_hand | mcp_joint | leap-hand/LEAP_Hand_Sim@master:assets/leap_hand/mcp_joint.stl | m->mm | [MIT](https://github.com/leap-hand/LEAP_Hand_Sim/blob/master/LICENSE.txt) |
| leap_hand | dip | leap-hand/LEAP_Hand_Sim@master:assets/leap_hand/dip.stl | m->mm | [MIT](https://github.com/leap-hand/LEAP_Hand_Sim/blob/master/LICENSE.txt) |
| leap_hand | pip | leap-hand/LEAP_Hand_Sim@master:assets/leap_hand/pip.stl | m->mm | [MIT](https://github.com/leap-hand/LEAP_Hand_Sim/blob/master/LICENSE.txt) |
| aloha_vx300s | gripper_finger | Interbotix/interbotix_ros_manipulators@main:interbotix_ros_xsarms/interbotix_xsarm_descriptions/meshes/vx300s_meshes/vx300s_10_gripper_finger.stl | mm | [BSD-3-Clause](https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/LICENSE) |
| xarm_gripper | left_finger | xArm-Developer/xarm_ros@master:xarm_description/meshes/gripper/xarm/left_finger.stl | m->mm | [BSD-3-Clause](https://github.com/xArm-Developer/xarm_ros/blob/master/LICENSE) |
| xarm_gripper | right_finger | xArm-Developer/xarm_ros@master:xarm_description/meshes/gripper/xarm/right_finger.stl | m->mm | [BSD-3-Clause](https://github.com/xArm-Developer/xarm_ros/blob/master/LICENSE) |
| xarm_gripper | lite_finger | xArm-Developer/xarm_ros@master:xarm_description/meshes/gripper/lite/visual/finger1.stl | m->mm | [BSD-3-Clause](https://github.com/xArm-Developer/xarm_ros/blob/master/LICENSE) |
| xarm_gripper | bio_left_finger | xArm-Developer/xarm_ros@master:xarm_description/meshes/gripper/bio/left_finger.stl | m->mm | [BSD-3-Clause](https://github.com/xArm-Developer/xarm_ros/blob/master/LICENSE) |
| onrobot_rg2 | inner_finger | Osaka-University-Harada-Laboratory/onrobot@main:onrobot_rg_description/meshes/rg2/visual/inner_finger.stl | m->mm | [MIT](https://github.com/Osaka-University-Harada-Laboratory/onrobot/blob/main/LICENSE) |
| onrobot_rg6 | inner_finger | Osaka-University-Harada-Laboratory/onrobot@main:onrobot_rg_description/meshes/rg6/visual/inner_finger.stl | m->mm | [MIT](https://github.com/Osaka-University-Harada-Laboratory/onrobot/blob/main/LICENSE) |
| onrobot_vg10 | suction_cups | Osaka-University-Harada-Laboratory/onrobot@main:onrobot_vg_description/meshes/vg10/visual/suction_cups.stl | m->mm | [MIT](https://github.com/Osaka-University-Harada-Laboratory/onrobot/blob/main/LICENSE) |
| onrobot_vg10 | base | Osaka-University-Harada-Laboratory/onrobot@main:onrobot_vg_description/meshes/vg10/visual/base_link.stl | m->mm | [MIT](https://github.com/Osaka-University-Harada-Laboratory/onrobot/blob/main/LICENSE) |
| schunk_sdh | finger_tip | ipa320/schunk_modular_robotics@indigo_dev:schunk_description/meshes/sdh/finger_tip.stl | m->mm | [Apache-2.0](https://github.com/ipa320/schunk_modular_robotics/blob/indigo_dev/schunk_description/package.xml) (package.xml) |
| schunk_sdh | finger | ipa320/schunk_modular_robotics@indigo_dev:schunk_description/meshes/sdh/finger.stl | m->mm | [Apache-2.0](https://github.com/ipa320/schunk_modular_robotics/blob/indigo_dev/schunk_description/package.xml) (package.xml) |
| schunk_sdh | palm | ipa320/schunk_modular_robotics@indigo_dev:schunk_description/meshes/sdh/palm.stl | m->mm | [Apache-2.0](https://github.com/ipa320/schunk_modular_robotics/blob/indigo_dev/schunk_description/package.xml) (package.xml) |
| unitree_go2 | foot | unitreerobotics/unitree_ros@master:robots/go2_description/meshes/foot.dae | m->mm | [BSD-3-Clause](https://github.com/unitreerobotics/unitree_ros/blob/master/LICENSE) |
| unitree_go2 | calf | unitreerobotics/unitree_ros@master:robots/go2_description/meshes/calf.dae | m->mm | [BSD-3-Clause](https://github.com/unitreerobotics/unitree_ros/blob/master/LICENSE) |
| allegro | fingertip | simlabrobotics/allegro_hand_ros@master:allegro_hand_description/meshes/link_3.0_tip.STL | m->mm | [BSD](https://github.com/simlabrobotics/allegro_hand_ros/blob/master/allegro_hand_description/manifest.xml) (ROS manifest.xml only; no LICENSE file) |
| allegro | thumb_fingertip | simlabrobotics/allegro_hand_ros@master:allegro_hand_description/meshes/link_15.0_tip.STL | m->mm | [BSD](https://github.com/simlabrobotics/allegro_hand_ros/blob/master/allegro_hand_description/manifest.xml) (ROS manifest.xml only; no LICENSE file) |
| allegro | palm | simlabrobotics/allegro_hand_ros@master:allegro_hand_description/meshes/base_link.STL | m->mm | [BSD](https://github.com/simlabrobotics/allegro_hand_ros/blob/master/allegro_hand_description/manifest.xml) (ROS manifest.xml only; no LICENSE file) |
| orca_hand | fingertip | orcahand/orcahand_description@main:assets/mjcf/left/I-FingerTipAssembly.stl | mm | [MIT](https://github.com/orcahand/orcahand_description/blob/main/LICENSE) |
| orca_hand | palm | orcahand/orcahand_description@main:assets/mjcf/left/L-Carpals.stl | mm | [MIT](https://github.com/orcahand/orcahand_description/blob/main/LICENSE) |
| yale_openhand | model_o_distal | grablab/openhand_simulation@master:model_o/distal.STL | mm | none stated in the repo; the same hands in [grablab/openhand-hardware](https://github.com/grablab/openhand-hardware/blob/master/LICENSE.md) are CC BY-NC 3.0 |
| yale_openhand | model_o_proximal | grablab/openhand_simulation@master:model_o/proximal.STL | mm | none stated in the repo; the same hands in [grablab/openhand-hardware](https://github.com/grablab/openhand-hardware/blob/master/LICENSE.md) are CC BY-NC 3.0 |
| unitree_b2 | foot | unitreerobotics/unitree_ros@master:robots/b2_description_mujoco/meshes/FL_foot.obj | m->mm | [BSD-3-Clause](https://github.com/unitreerobotics/unitree_ros/blob/master/LICENSE) |
| inspire_hand | index_fingertip | EuncheolIm/Inspire_hand_description@master:Inspire_hand/visual/right_index_intermediate_visual.obj | m->mm | [CC BY-NC-SA 4.0](https://github.com/EuncheolIm/Inspire_hand_description#license--attribution) (README, after [dex-urdf](https://github.com/dexsuite/dex-urdf); no LICENSE file) |
| inspire_hand | thumb_distal | EuncheolIm/Inspire_hand_description@master:Inspire_hand/visual/right_thumb_distal_visual.obj | m->mm | [CC BY-NC-SA 4.0](https://github.com/EuncheolIm/Inspire_hand_description#license--attribution) (README, after [dex-urdf](https://github.com/dexsuite/dex-urdf); no LICENSE file) |
| inspire_hand | palm | EuncheolIm/Inspire_hand_description@master:Inspire_hand/visual/right_base_link_visual.obj | m->mm | [CC BY-NC-SA 4.0](https://github.com/EuncheolIm/Inspire_hand_description#license--attribution) (README, after [dex-urdf](https://github.com/dexsuite/dex-urdf); no LICENSE file) |
| yale_openhand | t42_distal | grablab/openhand_simulation@master:t42/distal.stl | x25.4 (inches) | none stated in the repo; the same hands in [grablab/openhand-hardware](https://github.com/grablab/openhand-hardware/blob/master/LICENSE.md) are CC BY-NC 3.0 |
| yale_openhand | t42_proximal | grablab/openhand_simulation@master:t42/proximal.stl | x25.4 (inches) | none stated in the repo; the same hands in [grablab/openhand-hardware](https://github.com/grablab/openhand-hardware/blob/master/LICENSE.md) are CC BY-NC 3.0 |
| umi | soft_gripper_finger | actuated-umi/actuated-umi-gripper@main:3d-printables/actuated-UMI-v1/UMI-gripper-soft-gripper-finger.stl | mm | [MIT](https://github.com/actuated-umi/actuated-umi-gripper/blob/main/LICENSE) |
| umi | jaw_left | actuated-umi/actuated-umi-gripper@main:3d-printables/actuated-UMI-v2/actuated-UMI-jaw-left.stl | mm | [MIT](https://github.com/actuated-umi/actuated-umi-gripper/blob/main/LICENSE) |
| umi | jaw_right | actuated-umi/actuated-umi-gripper@main:3d-printables/actuated-UMI-v2/actuated-UMI-jaw-right.stl | mm | [MIT](https://github.com/actuated-umi/actuated-umi-gripper/blob/main/LICENSE) |
| agilex_piper | jaw_link7 | agilexrobotics/piper_ros@noetic:src/piper_description/meshes/link7.STL | m->mm | [MIT](https://github.com/agilexrobotics/piper_ros/blob/noetic/LICENSE) |
| agilex_piper | jaw_link8 | agilexrobotics/piper_ros@noetic:src/piper_description/meshes/link8.STL | m->mm | [MIT](https://github.com/agilexrobotics/piper_ros/blob/noetic/LICENSE) |
| agilex_piper | gripper_base | agilexrobotics/piper_ros@noetic:src/piper_description/meshes/gripper_base.STL | m->mm | [MIT](https://github.com/agilexrobotics/piper_ros/blob/noetic/LICENSE) |
| unitree_dex3 | distal_finger | unitreerobotics/unitree_ros@master:robots/dexterous_hand_description/dex3_1/meshes/right_hand_index_1_link.STL | m->mm | [BSD-3-Clause](https://github.com/unitreerobotics/unitree_ros/blob/master/LICENSE) |
| unitree_dex3 | thumb_distal | unitreerobotics/unitree_ros@master:robots/dexterous_hand_description/dex3_1/meshes/right_hand_thumb_2_link.STL | m->mm | [BSD-3-Clause](https://github.com/unitreerobotics/unitree_ros/blob/master/LICENSE) |
| unitree_dex3 | palm | unitreerobotics/unitree_ros@master:robots/dexterous_hand_description/dex3_1/meshes/right_hand_palm_link.STL | m->mm | [BSD-3-Clause](https://github.com/unitreerobotics/unitree_ros/blob/master/LICENSE) |
| dclaw | fingertip | google-research/robel-scenes@master:dclaw/meshes/xh_finger_tip.stl | mm | [Apache-2.0](https://github.com/google-research/robel-scenes/blob/master/LICENSE) |
| dclaw | fingertip_4 | google-research/robel-scenes@master:dclaw/meshes/xh_4_finger_tip.stl | mm | [Apache-2.0](https://github.com/google-research/robel-scenes/blob/master/LICENSE) |

Files directly under `examples/stl/` (not from this table):
`Moving_Jaw_08d.stl` and `Wrist_Roll_08c.stl` carry the names of SO-100
follower parts in TheRobotStudio/SO-ARM100 (Apache-2.0);
`FINGER PAD_t30_L.stl` (Yubi gripper finger pad) and `finger_tip_soft.stl`
have no recorded source.

Note on UMI: the original Universal Manipulation Interface publishes its
gripper CAD on Onshape rather than as STLs in a git repo, so these come from
the actuated-UMI project, which reuses UMI's soft gripper finger. The
`soft_gripper_finger` part is the UMI contact surface itself; the two jaws are
from the actuated variant. If you have the original UMI STLs, drop them in and
they take precedence. The actuated-UMI repository also carries UMI's MIT
license (Columbia Artificial Intelligence and Robotics Lab) in its
LICENSE_THIRD_PARTY file.
