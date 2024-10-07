import os, sys
import yaml
import torch
import tyro
import viser
import imageio
import numpy as onp
import joblib
import time
from scipy.spatial.transform import Rotation as R
from loguru import logger
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_matrix

from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda

    

def load_slam_results(slam_results_pt: str, return_y_up: bool = True):
    slam_results = torch.load(slam_results_pt)

    # Move slam results to the world coordinate
    slam_cam_position = slam_results[:, :3] # xyz

    slam_cam_quat = slam_results[:, 3:] # xyzw
    slam_cam_quat = slam_cam_quat[:, [3, 0, 1, 2]]
    slam_cam_matrix = quaternion_to_matrix(torch.tensor(slam_cam_quat)).float()
    slam_cam_position = torch.tensor(slam_cam_position).float()
    
    if return_y_up:
        yup2ydown = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()
        
        slam_cam_matrix = yup2ydown.mT @ slam_cam_matrix
        slam_cam_position = (yup2ydown.mT @ slam_cam_position.unsqueeze(-1)).squeeze(-1)
                

    return slam_cam_position, slam_cam_matrix

def main(sid: int = 0, result_pt: str = None):
    pred = torch.load(result_pt)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    smpl_faces = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()


    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
    global_verts = pred_ay_verts.cpu().numpy()

    # get camera orientation and position
    # get the first 3x3 matrix from the pred_smpl_params_global
    glob_ori = pred['smpl_params_global']['global_orient']
    cam_ori = pred['smpl_params_incam']['global_orient']
    glob_ori = axis_angle_to_matrix(glob_ori) # (N, 3, 3)
    cam_ori = axis_angle_to_matrix(cam_ori) # (N, 3, 3)
    cam_axes_in_world = glob_ori @ cam_ori.mT

    glob_transl = pred['smpl_params_global']['transl']
    cam_transl = pred['smpl_params_incam']['transl']

    cam_origin_world = glob_transl - (cam_axes_in_world @ cam_transl.unsqueeze(-1)).squeeze(-1)

    # get slam camera orientation and position
    slam_results_pt = result_pt.replace("hmr4d_results.pt", "preprocess/slam_results.pt")
    slam_cam_origin_world, slam_cam_axes_in_world = load_slam_results(slam_results_pt, return_y_up=True)

    # set the ground to be the minimum y value
    ground_y = global_verts[..., 1].min()
    global_verts[..., 1] = global_verts[..., 1] - ground_y

    timesteps = len(global_verts)

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
    


    
    # match the y level
    cam_origin_world[..., 1] = cam_origin_world[..., 1] - ground_y
    # slam_cam_origin_world[..., 1] = slam_cam_origin_world[..., 1] - ground_y

    # scale of DVPO is unknown
    # from slam_cam_origin_world, find an index where the 3d coordinate changes
    # then, use the scale of the first frame to scale the rest of the frames
    diff_idx_list = [0]
    for i in range(1, len(slam_cam_origin_world)):
        if onp.linalg.norm(slam_cam_origin_world[i] - slam_cam_origin_world[i-1]) > 1e-3:
            diff_idx_list.append(i)

    # average the scale o
    scale_list = []        
    for i in range(1, len(diff_idx_list)):
        scale = onp.linalg.norm(cam_origin_world[diff_idx_list[i-1]] - cam_origin_world[diff_idx_list[i]]) / onp.linalg.norm(slam_cam_origin_world[diff_idx_list[i-1]] - slam_cam_origin_world[diff_idx_list[i]])
        scale_list.append(scale)

    scale = onp.mean(scale_list, axis=0, keepdims=True)

    slam_cam_origin_world = slam_cam_origin_world * scale
    slam_cam_origin_world = slam_cam_origin_world - slam_cam_origin_world[0:1] + cam_origin_world[0:1]

    # trick scaling
    # cam_origin_world[..., 2] = cam_origin_world[..., 2] * 0.3
    # slam_cam_origin_world[..., 2] = slam_cam_origin_world[..., 2] * 0.3

   


    frame_nodes: list[viser.FrameHandle] = []
    for t in range(timesteps):
        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/t{t}", show_axes=False))

        server.scene.add_mesh_simple(
            f"/t{t}/mesh",
            vertices=onp.array(global_verts[t]),
            faces=onp.array(smpl_faces),
            flat_shading=False,
            wireframe=False,
        )
        
        cam_axes_matrix = cam_axes_in_world[t]
        cam_axes_quat =R.from_matrix(cam_axes_matrix).as_quat(scalar_first=True)
        
        server.scene.add_frame(
            f"/t{t}/cam",
            wxyz=cam_axes_quat,
            position=cam_origin_world[t],
            show_axes=True,
            axes_length=0.5,
            axes_radius=0.04,
        )
        
        slam_cam_axes_matrix = slam_cam_axes_in_world[t]
        slam_cam_axes_quat =R.from_matrix(slam_cam_axes_matrix).as_quat(scalar_first=True)
        
        server.scene.add_frame(
            f"/t{t}/slam_cam",
            wxyz=slam_cam_axes_quat,
            position=slam_cam_origin_world[t],
            show_axes=True,
            axes_length=0.8,
            axes_radius=0.04,
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