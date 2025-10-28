"""
jarvis submit -c create_merlin_gpu.py
"""
######################## rh2 ########################
from easydict import EasyDict as edict

rh2 = edict()
rh2.name = "semantic_pcd_job"
rh2.entry_point_content = f"""
set -x

if [ -d /mnt/bn/semantic-pcd-yg ];then
    sudo ln -s /mnt/bn/semantic-pcd-yg /mnt/bn/picoroomplan
    sudo ln -s /mnt/bn/semantic-pcd-yg /mnt/bn/picoroomplan2
    sudo ln -s /mnt/bn/semantic-pcd-yg /mnt/bn/semantic-pcd
fi
source /mnt/bn/picoroomplan/wbw/env/photoid_env
source /mnt/bn/picoroomplan/wbw/env/dot_command
set_dev_env
sleep 1000000000m
"""
rh2.entry_point = "/opt/tiger/submit_code/rh2_entry_point.sh"
rh2.mnt = "/opt/tiger/submit_code"
rh2.args = ""
rh2.resource = edict()
rh2.resource.arnold_config = edict()
rh2.resource.arnold_config.bytenasVolumes = [
    # {"accessMode": "RW", "name": "semantic-pcd-yg", "roles": []},
    {"accessMode": "RW", "name": "picoroomplan", "roles": []},
    {"accessMode": "RW", "name": "picoroomplan2", "roles": []},
    {"accessMode": "RW", "name": "semantic-pcd", "roles": []},
]
# rh2.resource.arnold_config.group_names = ["pico-mr"]
rh2.resource.arnold_config.group_names = ["pico-ai"]
rh2.resource.arnold_config.cluster_name = "cloudnative-lq"
# rh2.resource.arnold_config.gpu_type =  "A100_SXM4_40GB"
rh2.resource.arnold_config.gpu_type = "A100_SXM_80GB"
# rh2.resource.arnold_config.cluster_name = "cloudnative-yg"
# rh2.resource.arnold_config.gpu_type =  "A800_SXM_40GB"
rh2.resource.arnold_config.roles = [
    {
        "name": "worker",
        "num": 1,
        "gpuv": rh2.resource.arnold_config.gpu_type,
        "gpu": 1,
        "cpu": 12,
        "memory": 1024 * 128,  # 128G
    }
]

rh2.icm_image = "hub.byted.org/base/lab.cuda.devel:12.3.2"  # https://bytedance.larkoffice.com/wiki/wikcn8eI9YbG9rBfmymBHt8I3BE
rh2.python_version = "PYTHON3"
rh2.apt = ["libgl1", "tmux", "htop", "universal-ctags", "libgtk2.0-0"]
rh2.pip3 = [
    "opencv-python",
    "easydict",
]
rh2.env = {"BYTED_TORCH_BYTECCL": "O0"}

# rh2_end
