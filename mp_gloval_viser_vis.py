import os, sys
import yaml
import torch
import tyro
import viser
import imageio
import numpy as onp
import joblib
import time
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from loguru import logger
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_matrix, matrix_to_axis_angle

from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda


# R_y_upsidedown = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()
# Different from WHAM's R_y_upsidedown


def get_color_for_sid(sid):
    # Simple hash function to generate a color
    hash_value = sid * 123456789 + 111111111
    r = (hash_value & 0xFF0000) >> 16
    g = (hash_value & 0x00FF00) >> 8
    b = hash_value & 0x0000FF
    return (r, g, b )
     

def main(sid: int = 0, result_pt: str = None):
    gvhmr_results = torch.load(result_pt)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    smpl_faces = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
    tracking_results = torch.load(result_pt.replace("hmr4d_results.pt", "preprocess/bbx.pt"))

    # match WHAM visualization 
    # 180 degrees rotation around y axis
    whamify_rotation_matrix = torch.tensor([
        [-1., 0., 0.],
        [0., 1., 0.],
        [0., 0., -1.]
    ])
    # change the world frame!!


    data_frames = defaultdict(dict)
    for tracking_id, tracking_data in tracking_results.items():
        frame_ids = tracking_data['frame_ids'].tolist()

        gvhmr_pred = gvhmr_results[tracking_id]

        # get camera orientation and position
        glob_ori = gvhmr_pred['smpl_params_global']['global_orient'] 
        cam_ori = gvhmr_pred['smpl_params_incam']['global_orient']
        glob_ori = axis_angle_to_matrix(glob_ori) # (N, 3, 3)
        cam_ori = axis_angle_to_matrix(cam_ori) # (N, 3, 3)
        cam_axes_in_world = glob_ori @ cam_ori.mT

        glob_transl = gvhmr_pred['smpl_params_global']['transl']
        cam_transl = gvhmr_pred['smpl_params_incam']['transl']
        cam_origin_world = glob_transl - (cam_axes_in_world @ cam_transl.unsqueeze(-1)).squeeze(-1)

        # whamify the camera axes
        cam_axes_in_world = whamify_rotation_matrix[None, :, :] @ cam_axes_in_world
        cam_origin_world = (whamify_rotation_matrix[None, :, :] @ cam_origin_world.unsqueeze(-1)).squeeze(-1)    
        # whamify the global orient and transl
        glob_ori = whamify_rotation_matrix[None, :, :] @ glob_ori
        glob_transl = (whamify_rotation_matrix[None, :, :] @ glob_transl.unsqueeze(-1)).squeeze(-1)
        
        gvhmr_pred['smpl_params_global']['global_orient'] = matrix_to_axis_angle(glob_ori)
        gvhmr_pred['smpl_params_global']['transl'] = glob_transl

        for i, frame_id in enumerate(frame_ids):
            data_frames[frame_id][tracking_id] = {
                'cam_axes': cam_axes_in_world[i].cpu().numpy(),
                'cam_origin': cam_origin_world[i].cpu().numpy(),

                'pose_world': gvhmr_pred['smpl_params_global']['global_orient'][i].cpu().numpy(),
                'trans_world': gvhmr_pred['smpl_params_global']['transl'][i].cpu().numpy(),
                
                'betas': gvhmr_pred['smpl_params_global']['betas'][i].cpu().numpy(),
                'body_pose_world': gvhmr_pred['smpl_params_global']['body_pose'][i].cpu().numpy(),
            }

    # Make sure the frame_id starts from 0 
    min_frame_id = min(data_frames.keys())
    new_data_frames = defaultdict(dict)
    for frame_id in data_frames.keys():
        new_data_frames[frame_id - min_frame_id] = data_frames[frame_id]
    data_frames = new_data_frames

    global_y_min_list = []

    prev_first_sid = min(list(data_frames[0].keys()))
    ref_sid_cam_origin = prev_first_sid_cam_origin = data_frames[0][prev_first_sid]['cam_origin']

    # integrate everyone to one global coordinate per frame
    for frame_id in sorted(data_frames.keys()):
        first_sid = min(list(data_frames[frame_id].keys()))
        first_sid_cam_axes = data_frames[frame_id][first_sid]['cam_axes']
        first_sid_cam_origin = data_frames[frame_id][first_sid]['cam_origin']


        # handle the abrupt change of camera origin
        if first_sid != prev_first_sid:
            ref_sid_cam_origin = prev_first_sid_cam_origin + 1 * cam_origin_vel # assuming frame_id change only for one time frame.
            prev_first_sid_cam_origin = first_sid_cam_origin
            prev_first_sid = first_sid
        else:
            cam_origin_vel = first_sid_cam_origin - prev_first_sid_cam_origin
            ref_sid_cam_origin = ref_sid_cam_origin + 1 * cam_origin_vel # assuming frame_id change only for one time frame.
            prev_first_sid_cam_origin = first_sid_cam_origin
            prev_first_sid = first_sid

        # TODO: handle the abrupt change of camera axes
        ref_sid_cam_axes = first_sid_cam_axes
        

        for sid in data_frames[frame_id].keys():
            sid_cam_axes = data_frames[frame_id][sid]['cam_axes']
            sid_cam_origin = data_frames[frame_id][sid]['cam_origin']

            sid_global_orient = data_frames[frame_id][sid]['pose_world']
            sid_trans_world = data_frames[frame_id][sid]['trans_world']
            sid_betas = data_frames[frame_id][sid]['betas']
            sid_body_pose = data_frames[frame_id][sid]['body_pose_world']
        
            sid_global_orient = R.from_rotvec(sid_global_orient).as_matrix()

            # compute relative transformation from sid to first_sid
            # rel_cam_rot = first_sid_cam_axes @ sid_cam_axes.T
            # rel_cam_transl = first_sid_cam_origin - sid_cam_origin
            rel_cam_rot = ref_sid_cam_axes @ sid_cam_axes.T
            rel_cam_transl = ref_sid_cam_origin - sid_cam_origin

            sid_global_orient = rel_cam_rot @ sid_global_orient
            sid_trans_world = rel_cam_transl + sid_trans_world

            # compute global vertices
            sid_global_orient = R.from_matrix(sid_global_orient).as_rotvec()

            # smpl
            smpl_params_global = {
                'global_orient': torch.from_numpy(sid_global_orient).float().unsqueeze(0),
                'transl': torch.from_numpy(sid_trans_world).float().unsqueeze(0),
                'betas': torch.from_numpy(sid_betas).float().unsqueeze(0),
                'body_pose': torch.from_numpy(sid_body_pose).float().unsqueeze(0),
            }

            smplx_out = smplx(**to_cuda(smpl_params_global))
            pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
            global_verts = pred_ay_verts.cpu().numpy()

            data_frames[frame_id][sid]['global_verts'] = global_verts
            global_y_min = data_frames[frame_id][sid]['global_verts'][..., 1].min()
            global_y_min_list.append(global_y_min)
            
        data_frames[frame_id]['ref_cam_axes'] = ref_sid_cam_axes # first_sid_cam_axes
        data_frames[frame_id]['ref_cam_origin'] = ref_sid_cam_origin # first_sid_cam_origin

    print("Integrated all frames into one global coordinate!")
    # pick groun_y from the first frame
    ground_y = global_y_min_list[0]

    timesteps = max(data_frames.keys()) - min(data_frames.keys()) + 1

    # setup viser server
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.add_grid("ground", width=35, height=35, cell_size=1, plane="xz")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        print("new client!")

        client.camera.position = onp.array([1.14120013, 0.60690449, 5.17581808]) # onp.array([-1, 4, 13])
        client.camera.wxyz = onp.array([-1.75483266e-01,  9.83732196e-01 , 4.88596244e-04, 3.84233121e-02])
            
        # # This will run whenever we get a new camera!
        # @client.camera.on_update
        # def _(_: viser.CameraHandle) -> None:
        #     print(f"New camera on client {client.client_id}!")
        #     print(f"Camera pose for client {id}")
        #     print(f"\tfov: {client.camera.fov}")
        #     print(f"\taspect: {client.camera.aspect}")
        #     print(f"\tlast update: {client.camera.update_timestamp}")
        #     print(f"\twxyz: {client.camera.wxyz}")
        #     print(f"\tposition: {client.camera.position}")
        #     print(f"\tlookat: {client.camera.look_at}")
            
        # Show the client ID in the GUI.
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True
    

    frame_nodes: list[viser.FrameHandle] = []
    for t in range(timesteps):
        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/t{t}", show_axes=False))
        
        cam_axes_matrix = data_frames[t]['ref_cam_axes']
        cam_axes_quat = R.from_matrix(cam_axes_matrix).as_quat(scalar_first=True)
        cam_origin = data_frames[t]['ref_cam_origin'] - ground_y
        server.scene.add_frame(
            f"/t{t}/cam",
            wxyz=cam_axes_quat,
            position=cam_origin,
            show_axes=True,
            axes_length=0.5,
            axes_radius=0.04,
        )

        for sid in data_frames[t].keys():
            if sid == 'ref_cam_axes' or sid == 'ref_cam_origin':
                continue

            global_verts = data_frames[t][sid]['global_verts'] - ground_y
            server.scene.add_mesh_simple(
                f"/t{t}/mesh{sid}",
                vertices=onp.array(global_verts),
                faces=onp.array(smpl_faces),
                flat_shading=False,
                wireframe=False,
                color=get_color_for_sid(sid),
            )

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=timesteps - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=15
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % timesteps

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % timesteps

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    render_button = server.gui.add_button("Render motion", disabled=False)
    recording = False
    @render_button.on_click
    def _(event: viser.GuiEvent) -> None:
        nonlocal recording
     
        client = event.client
        if not recording:
            recording = True
            gui_playing.value = False
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value
            gui_framerate.disabled = False
            
            # images = []
            writer = imageio.get_writer(
                'output.mp4', 
                fps=gui_framerate.value, mode='I', format='FFMPEG', macro_block_size=1
            )
            while True:
                if recording:
                    gui_timestep.value = (gui_timestep.value + 1) % timesteps
                    # images.append(client.camera.get_render(height=720, width=1280))
                    img = client.camera.get_render(height=480, width=720)
                    writer.append_data(img)
                    print('recording...')
                else:
                    print("Recording stopped")
                    gui_framerate.disabled = True
                    writer.close()
                    break
        else:
            recording = False

        

        
    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % timesteps

        time.sleep(1.0 / gui_framerate.value)

if __name__ == "__main__":
    tyro.cli(main)