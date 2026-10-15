# End-effector mesh sources

Converted to binary STL in millimetres via trimesh (metres-scale
sources multiplied by 1000). Check each repo's license before
redistributing beyond this project.

| robot | part | source (repo@branch:path) | units |
|---|---|---|---|
| so101 | moving_jaw | TheRobotStudio/SO-ARM100@main:STL/SO101/Individual/Moving_Jaw_SO101.stl | mm |
| so101 | fixed_jaw_wrist_roll | TheRobotStudio/SO-ARM100@main:STL/SO101/Individual/Wrist_Roll_Follower_SO101.stl | mm |
| robotiq_2f85 | pad | ros-industrial/robotiq@kinetic-devel:robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_pad.dae | mm |
| robotiq_2f85 | inner_finger | ros-industrial/robotiq@kinetic-devel:robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_inner_finger.dae | mm |
| robotiq_2f140 | inner_finger | ros-industrial/robotiq@kinetic-devel:robotiq_2f_140_gripper_visualization/meshes/visual/robotiq_arg2f_140_inner_finger.stl | m->mm |
| franka | fingertip | frankaemika/franka_description@main:meshes/robot_ee/franka_hand_white/visual/finger.dae | m->mm |
| franka | hand_body | frankaemika/franka_description@main:meshes/robot_ee/franka_hand_white/collision/hand.stl | m->mm |
| franka | arm_link4 | frankaemika/franka_description@main:meshes/robots/fer/visual/link4.dae | m->mm |
| franka | arm_link5 | frankaemika/franka_description@main:meshes/robots/fer/visual/link5.dae | m->mm |
| leap_hand | fingertip | leap-hand/LEAP_Hand_Sim@master:assets/leap_hand/fingertip.stl | m->mm |
| leap_hand | thumb_fingertip | leap-hand/LEAP_Hand_Sim@master:assets/leap_hand/thumb_fingertip.stl | m->mm |
| leap_hand | palm | leap-hand/LEAP_Hand_Sim@master:assets/leap_hand/palm_lower.stl | m->mm |
| leap_hand | mcp_joint | leap-hand/LEAP_Hand_Sim@master:assets/leap_hand/mcp_joint.stl | m->mm |
| leap_hand | dip | leap-hand/LEAP_Hand_Sim@master:assets/leap_hand/dip.stl | m->mm |
| leap_hand | pip | leap-hand/LEAP_Hand_Sim@master:assets/leap_hand/pip.stl | m->mm |
| aloha_vx300s | gripper_finger | Interbotix/interbotix_ros_manipulators@main:interbotix_ros_xsarms/interbotix_xsarm_descriptions/meshes/vx300s_meshes/vx300s_10_gripper_finger.stl | mm |
| xarm_gripper | left_finger | xArm-Developer/xarm_ros@master:xarm_description/meshes/gripper/xarm/left_finger.stl | m->mm |
| xarm_gripper | right_finger | xArm-Developer/xarm_ros@master:xarm_description/meshes/gripper/xarm/right_finger.stl | m->mm |
| xarm_gripper | lite_finger | xArm-Developer/xarm_ros@master:xarm_description/meshes/gripper/lite/visual/finger1.stl | m->mm |
| xarm_gripper | bio_left_finger | xArm-Developer/xarm_ros@master:xarm_description/meshes/gripper/bio/left_finger.stl | m->mm |
| onrobot_rg2 | inner_finger | Osaka-University-Harada-Laboratory/onrobot@main:onrobot_rg_description/meshes/rg2/visual/inner_finger.stl | m->mm |
| onrobot_rg6 | inner_finger | Osaka-University-Harada-Laboratory/onrobot@main:onrobot_rg_description/meshes/rg6/visual/inner_finger.stl | m->mm |
| onrobot_vg10 | suction_cups | Osaka-University-Harada-Laboratory/onrobot@main:onrobot_vg_description/meshes/vg10/visual/suction_cups.stl | m->mm |
| onrobot_vg10 | base | Osaka-University-Harada-Laboratory/onrobot@main:onrobot_vg_description/meshes/vg10/visual/base_link.stl | m->mm |
| schunk_sdh | finger_tip | ipa320/schunk_modular_robotics@indigo_dev:schunk_description/meshes/sdh/finger_tip.stl | m->mm |
| schunk_sdh | finger | ipa320/schunk_modular_robotics@indigo_dev:schunk_description/meshes/sdh/finger.stl | m->mm |
| schunk_sdh | palm | ipa320/schunk_modular_robotics@indigo_dev:schunk_description/meshes/sdh/palm.stl | m->mm |
| unitree_go2 | foot | unitreerobotics/unitree_ros@master:robots/go2_description/meshes/foot.dae | m->mm |
| unitree_go2 | calf | unitreerobotics/unitree_ros@master:robots/go2_description/meshes/calf.dae | m->mm |
| allegro | fingertip | simlabrobotics/allegro_hand_ros@master:allegro_hand_description/meshes/link_3.0_tip.STL | m->mm |
| allegro | thumb_fingertip | simlabrobotics/allegro_hand_ros@master:allegro_hand_description/meshes/link_15.0_tip.STL | m->mm |
| allegro | palm | simlabrobotics/allegro_hand_ros@master:allegro_hand_description/meshes/base_link.STL | m->mm |
| orca_hand | fingertip | orcahand/orcahand_description@main:assets/mjcf/left/I-FingerTipAssembly.stl | mm |
| orca_hand | palm | orcahand/orcahand_description@main:assets/mjcf/left/L-Carpals.stl | mm |
| yale_openhand | model_o_distal | grablab/openhand_simulation@master:model_o/distal.STL | mm |
| yale_openhand | model_o_proximal | grablab/openhand_simulation@master:model_o/proximal.STL | mm |
| yale_openhand | t42_distal | grablab/openhand_simulation@master:t42/distal.stl | AMBIGUOUS UNITS |
| unitree_b2 | foot | unitreerobotics/unitree_ros@master:robots/b2_description_mujoco/meshes/FL_foot.obj | m->mm |
| inspire_hand | index_fingertip | EuncheolIm/Inspire_hand_description@master:Inspire_hand/visual/right_index_intermediate_visual.obj | m->mm |
| inspire_hand | thumb_distal | EuncheolIm/Inspire_hand_description@master:Inspire_hand/visual/right_thumb_distal_visual.obj | m->mm |
| inspire_hand | palm | EuncheolIm/Inspire_hand_description@master:Inspire_hand/visual/right_base_link_visual.obj | m->mm |
| yale_openhand | t42_distal | grablab/openhand_simulation@master:t42/distal.stl | x25.4 (inches) |
| yale_openhand | t42_proximal | grablab/openhand_simulation@master:t42/proximal.stl | x25.4 (inches) |
| umi | soft_gripper_finger | actuated-umi/actuated-umi-gripper@main:3d-printables/actuated-UMI-v1/UMI-gripper-soft-gripper-finger.stl | mm |
| umi | jaw_left | actuated-umi/actuated-umi-gripper@main:3d-printables/actuated-UMI-v2/actuated-UMI-jaw-left.stl | mm |
| umi | jaw_right | actuated-umi/actuated-umi-gripper@main:3d-printables/actuated-UMI-v2/actuated-UMI-jaw-right.stl | mm |
| agilex_piper | jaw_link7 | agilexrobotics/piper_ros@noetic:src/piper_description/meshes/link7.STL | m->mm |
| agilex_piper | jaw_link8 | agilexrobotics/piper_ros@noetic:src/piper_description/meshes/link8.STL | m->mm |
| agilex_piper | gripper_base | agilexrobotics/piper_ros@noetic:src/piper_description/meshes/gripper_base.STL | m->mm |
| unitree_dex3 | distal_finger | unitreerobotics/unitree_ros@master:robots/dexterous_hand_description/dex3_1/meshes/right_hand_index_1_link.STL | m->mm |
| unitree_dex3 | thumb_distal | unitreerobotics/unitree_ros@master:robots/dexterous_hand_description/dex3_1/meshes/right_hand_thumb_2_link.STL | m->mm |
| unitree_dex3 | palm | unitreerobotics/unitree_ros@master:robots/dexterous_hand_description/dex3_1/meshes/right_hand_palm_link.STL | m->mm |
| dclaw | fingertip | google-research/robel-scenes@master:dclaw/meshes/xh_finger_tip.stl | mm |
| dclaw | fingertip_4 | google-research/robel-scenes@master:dclaw/meshes/xh_4_finger_tip.stl | mm |

Note on UMI: the original Universal Manipulation Interface publishes its
gripper CAD on Onshape rather than as STLs in a git repo, so these come from
the actuated-UMI project, which reuses UMI's soft gripper finger. The
`soft_gripper_finger` part is the UMI contact surface itself; the two jaws are
from the actuated variant. If you have the original UMI STLs, drop them in and
they take precedence.
