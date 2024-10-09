import cv2
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from collections import defaultdict
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel
from hmr4d.utils.preproc.vitfeat_extractor import get_batch

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from tqdm import tqdm
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange


def get_color_for_sid(sid):
    # Simple hash function to generate a color
    hash_value = sid * 123456789 + 111111111
    r = (hash_value & 0xFF0000) >> 16
    g = (hash_value & 0x00FF00) >> 8
    b = hash_value & 0x0000FF
    return (r / 255, g / 255, b / 255)

CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    args = parser.parse_args()

    # Input
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")
    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
        ]

        # Allow to change output root
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=CRF)
        for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc=f"Copy"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    return cfg


@torch.no_grad()
def run_preprocess(cfg):
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        tracking_results = {}
        tracker = Tracker()
        multi_bbx_xyxy = tracker.get_multi_track(video_path)
        
        for tracking_id, tracking_data in multi_bbx_xyxy.items():
            bbx_xyxy = tracking_data['bbx_xyxy'].float()  # (L, 4)
            bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3) apply aspect ratio and enlarge
            tracking_results[tracking_id] = {
                'frame_ids': tracking_data['frame_ids'],
                'bbx_xyxy': bbx_xyxy,
                'bbx_xys': bbx_xys  
            }
        torch.save(tracking_results, paths.bbx)
        del tracker
    else:
        tracking_results = torch.load(paths.bbx)
        Log.info(f"[Preprocess] tracking results (frame_ids, bbx_xyxy, bbx_xys) from {paths.bbx}")


    if verbose:
        video = read_video_np(video_path)

        def draw_bbx_with_frame_ids(video, tracking_results):
            """
            Draw bounding boxes with frame_ids information for each subject in the video.
            
            Args:
                video (np.ndarray): Video frames of shape (num_frames, height, width, 3)
                tracking_results (dict): Dictionary containing tracking data for each subject
            
            Returns:
                np.ndarray: Video frames with bounding boxes and frame_ids drawn
            """
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            new_video = []
            for frame_idx, frame in enumerate(video):
                for tracking_id, tracking_data in tracking_results.items():
                    frame_ids = tracking_data['frame_ids']
                    bbx_xyxy = tracking_data['bbx_xyxy']
                    
                    if frame_idx in frame_ids:
                        idx = (frame_ids == frame_idx).nonzero().item()
                        bbox = bbx_xyxy[idx].int().tolist()
                        color = get_color_for_sid(tracking_id)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        
                        # Draw tracking ID and frame index
                        text = f"ID: {tracking_id}, Frame: {frame_idx}"
                        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                        cv2.rectangle(frame, (bbox[0], bbox[1] - text_size[1] - 5), (bbox[0] + text_size[0], bbox[1]), color, -1)
                        cv2.putText(frame, text, (bbox[0], bbox[1] - 5), font, font_scale, (255, 255, 255), font_thickness)

                new_video.append(frame)
            return new_video

        # Draw bounding boxes with frame_ids information
        video_with_bbx_and_ids = draw_bbx_with_frame_ids(video, tracking_results)
        save_video(video_with_bbx_and_ids, cfg.paths.bbx_xyxy_video_overlay)
        print("Saved yolo trackingvideo to ", cfg.paths.bbx_xyxy_video_overlay)


    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()

        vitpose = {}
        for tracking_id, tracking_data in tracking_results.items():
            bbx_xys = tracking_data['bbx_xys']
            frame_ids = tracking_data['frame_ids'].tolist()

            sampled_video = read_video_np(video_path)[frame_ids]
            imgs, bbx_xys = get_batch(sampled_video, bbx_xys, img_ds=1, path_type="np")
            assert len(imgs) == len(bbx_xys)

            vitpose[tracking_id] = vitpose_extractor.extract(imgs, bbx_xys)
            # vitpose[tracking_id] = vitpose_extractor.extract(video_path, bbx_xys)

        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")


    if verbose:
        from hmr4d.utils.vis.cv2_utils import draw_coco17_skeleton
        video = read_video_np(video_path)
        # for tracking_id, tracking_data in tracking_results.items():
        #     video = draw_coco17_skeleton_batch(video, vitpose[tracking_id], 0.5)
        def draw_coco17_skeleton_with_ids(video, vitpose, tracking_results):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            new_video = []
            for frame_idx, frame in enumerate(video):
                for tracking_id, tracking_data in tracking_results.items():
                    frame_ids = tracking_data['frame_ids']

                    if frame_idx in frame_ids:  
                        idx = (frame_ids == frame_idx).nonzero().item()
                        keypoints = vitpose[tracking_id][idx].cpu().numpy()
                        
                        # Draw skeleton
                        frame = draw_coco17_skeleton(frame, keypoints, conf_thr=0.5)
                        
                        # Draw tracking ID
                        text = f"ID: {tracking_id}"
                        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                        text_x = int(keypoints[0, 0])  # Use x-coordinate of first keypoint
                        text_y = int(keypoints[0, 1]) - 10  # Use y-coordinate of first keypoint, slightly above
                        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y), get_color_for_sid(tracking_id), -1)
                        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

                new_video.append(frame)
            return new_video

        # Draw coco17 skeleton with tracking IDs
        video_with_skeleton = draw_coco17_skeleton_with_ids(video, vitpose, tracking_results)
        save_video(video_with_skeleton, paths.vitpose_video_overlay)
        print("Saved vitpose video to ", paths.vitpose_video_overlay)
        import pdb; pdb.set_trace()

    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = {}
        # vitpose = {}
        # for tracking_id, tracking_data in tracking_results.items():
        #     bbx_xys = tracking_data['bbx_xys']
        #     frame_ids = tracking_data['frame_ids'].tolist()

        #     sampled_video = read_video_np(video_path)[frame_ids]
        #     imgs, bbx_xys = get_batch(sampled_video, bbx_xys, img_ds=1, path_type="np")
        #     assert len(imgs) == len(bbx_xys)

        #     vitpose[tracking_id] = vitpose_extractor.extract(imgs, bbx_xys)
        #     # vitpose[tracking_id] = vitpose_extractor.extract(video_path, bbx_xys)

        for tracking_id, tracking_data in tracking_results.items():
            bbx_xys = tracking_data['bbx_xys']
            frame_ids = tracking_data['frame_ids'].tolist()

            sampled_video = read_video_np(video_path)[frame_ids]
            imgs, bbx_xys = get_batch(sampled_video, bbx_xys, img_ds=1, path_type="np")
            assert len(imgs) == len(bbx_xys)

            vit_features[tracking_id] = extractor.extract_video_features(imgs, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        vit_features = torch.load(paths.vit_features)
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # Get DPVO results
    if not static_cam:  # use slam to get cam rotation
        if not Path(paths.slam).exists():
            length, width, height = get_video_lwh(cfg.video_path)
            K_fullimg = estimate_K(width, height)
            intrinsics = convert_K_to_K4(K_fullimg)
            slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
            bar = tqdm(total=length, desc="DPVO")
            while True:
                ret = slam.track()
                if ret:
                    bar.update()
                else:
                    break
            slam_results = slam.process()  # (L, 7), numpy
            torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
        R_w2c = quaternion_to_matrix(traj_quat).mT
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    # K_fullimg = create_camera_sensor(width, height, 26)[2].repeat(length, 1, 1)

    tracking_results = torch.load(paths.bbx)
    vitpose = torch.load(paths.vitpose)
    vit_features = torch.load(paths.vit_features)

    data_dict = {}
    for tracking_id, tracking_data in tracking_results.items():
        data = {}

        data['kp2d'] = vitpose[tracking_id]
        data['f_imgseq'] = vit_features[tracking_id]
        data['bbx_xys'] = tracking_data['bbx_xys']
        data['frame_ids'] = tracking_data['frame_ids']

        data['length'] = torch.tensor(len(data['frame_ids']))
        data['K_fullimg'] = K_fullimg[data['frame_ids']]
        data['cam_angvel'] = compute_cam_angvel(R_w2c[data['frame_ids']])

        assert len(data['frame_ids']) == len(data['kp2d']) == len(data['bbx_xys']) == len(data['f_imgseq']) == len(data['K_fullimg']) == len(data['cam_angvel']), \
            f"Data length mismatch for tracking_id {tracking_id}:\n" \
            f"frame_ids: {len(data['frame_ids'])}\n" \
            f"kp2d: {len(data['kp2d'])}\n" \
            f"bbx_xys: {len(data['bbx_xys'])}\n" \
            f"f_imgseq: {len(data['f_imgseq'])}\n" \
            f"K_fullimg: {len(data['K_fullimg'])}\n" \
            f"cam_angvel: {len(data['cam_angvel'])}"
        data_dict[tracking_id] = data


    # data = {
    #     "length": torch.tensor(length),
    #     "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
    #     "kp2d": torch.load(paths.vitpose),
    #     "K_fullimg": K_fullimg,
    #     "cam_angvel": compute_cam_angvel(R_w2c),
    #     "f_imgseq": torch.load(paths.vit_features),
    # }
    return data_dict


def render_incam(cfg):
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    total_pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    tracking_results = torch.load(cfg.paths.bbx)

    pred_c_verts_dict = defaultdict(dict) # {frame_id: {tracking_id: pred_c_verts}}
    for tracking_id, tracking_data in tracking_results.items():
        frame_ids = tracking_data['frame_ids'].tolist()

        pred = total_pred[tracking_id]
        smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
        pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

        for i, frame_id in enumerate(frame_ids):
            pred_c_verts_dict[frame_id][tracking_id] = pred_c_verts[i]

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    # bbx_xys_render = torch.load(cfg.paths.bbx)["bbx_xys"]

    # -- render mesh -- #
    writer = get_writer(incam_video_path, fps=30, crf=CRF)
    for frame_idx, img in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"):
        if frame_idx not in pred_c_verts_dict:
            continue
        print("Rendering frame_idx:", frame_idx)
        for tracking_id, pred_c_verts in pred_c_verts_dict[frame_idx].items():
            verts_incam = pred_c_verts
            img = renderer.render_mesh(verts_incam.cuda(), img, get_color_for_sid(tracking_id))

            # # bbx
            # bbx_xys_ = bbx_xys_render[i].cpu().numpy()
            # lu_point = (bbx_xys_[:2] - bbx_xys_[2:] / 2).astype(int)
            # rd_point = (bbx_xys_[:2] + bbx_xys_[2:] / 2).astype(int)
            # img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)
        writer.write_frame(img)

    writer.close()
    reader.close()


def render_global(cfg):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    def move_to_start_point_face_z(verts):
        "XZ to origin, Start from the ground, Face-Z"
        # position
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        # face direction
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    # renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    render_length = length if not debug_cam else 8
    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in tqdm(range(render_length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()


if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    # ===== Preprocess and save to disk ===== #
    run_preprocess(cfg)
    data_dict = load_data_dict(cfg)

    # ===== HMR4D ===== #
    total_pred = {}
    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        
        for tracking_id, data in data_dict.items():
            tic = Log.sync_time()
            pred = model.predict(data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
            total_pred[tracking_id] = pred
            data_time = data["length"] / 30
            Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
        torch.save(total_pred, paths.hmr4d_results)
    else:
        total_pred = torch.load(paths.hmr4d_results)
        Log.info(f"[HMR4D] Predicting results from {paths.hmr4d_results}")
    # ===== Render ===== #
    render_incam(cfg)
    # render_global(cfg)
    # if not Path(paths.incam_global_horiz_video).exists():
    #     Log.info("[Merge Videos]")
    #     merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)
