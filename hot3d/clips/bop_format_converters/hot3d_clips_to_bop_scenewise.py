# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script converts the Hot3D-Clips dataset used for the BOP challenge to the BOP format.
NOTE: the BOP format was updated from its classical format to a new format.
      The classical format had one main modality (rgb or gray) and depth.
      The new format can have multiple modalities (rgb, gray1, gray2) and no depth.
"""

import argparse
import json
import multiprocessing
import os
import sys
import tarfile

import cv2
import numpy as np
from bop_toolkit_lib import misc
from PIL import Image
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import clip_util


def main():
    # setup args
    parser = argparse.ArgumentParser()
    parser.add_argument("--hot3d-dataset-path", required=True, type=str)
    # BOP dataset split name
    parser.add_argument("--split", required=True, type=str)
    # number of threads
    parser.add_argument("--num-threads", type=int, default=4)

    args = parser.parse_args()

    # if split contains "quest3"
    if "quest3" in args.split:
        args.camera_streams_id = ["1201-1", "1201-2"]
        args.camera_streams_names = ["gray1", "gray2"]
    elif "aria" in args.split:
        args.camera_streams_id = ["214-1", "1201-1", "1201-2"]
        args.camera_streams_names = ["rgb", "gray1", "gray2"]
    else:
        print(
            "Split name must contain 'quest3' or 'aria'.\n"
            "Valid splits: train_quest3, test_quest3, train_aria, test_aria,\n"
            "object_ref_quest3_static, object_ref_quest3_dynamic,\n"
            "object_ref_aria_static, object_ref_aria_dynamic."
        )
        exit()

    # paths
    clips_input_dir = os.path.join(args.hot3d_dataset_path, args.split)
    scenes_output_dir = os.path.join(args.hot3d_dataset_path, args.split + "_scenewise")

    # list all clips names in the dataset
    split_clips = sorted([p for p in os.listdir(clips_input_dir) if p.endswith(".tar")])

    # create output directory
    os.makedirs(scenes_output_dir, exist_ok=True)

    # Progress bar setup
    with tqdm(total=len(split_clips), desc="Processing clips") as pbar:
        # Use a Pool of 8 processes
        with multiprocessing.Pool(processes=args.num_threads) as pool:
            # Use imap_unordered to get results as soon as they're ready
            for _ in pool.imap_unordered(
                worker,
                (
                    (clip, clips_input_dir, scenes_output_dir, args)
                    for clip in split_clips
                ),
            ):
                pbar.update(1)


def worker(args):
    clip, clips_input_dir, scenes_output_dir, args = args
    process_clip(clip, clips_input_dir, scenes_output_dir, args)


def detect_tar_prefix(tar):
    """Detect if tar members have a directory prefix (e.g. 'obj_000001_up/').

    Returns the prefix string (empty string if no prefix).
    """
    for name in tar.getnames():
        if name.endswith(".info.json"):
            # e.g. "obj_000001_up/000000.info.json" -> prefix = "obj_000001_up/"
            # or "000000.info.json" -> prefix = ""
            parts = name.rsplit("/", 1)
            if len(parts) == 2:
                return parts[0] + "/"
            return ""
    return ""


def get_number_of_frames_prefixed(tar, prefix):
    """Like clip_util.get_number_of_frames but handles a directory prefix."""
    max_frame_id = -1
    for x in tar.getnames():
        if x.endswith(".info.json"):
            basename = x[len(prefix):] if prefix and x.startswith(prefix) else x
            frame_id = int(basename.split(".info.json")[0])
            if frame_id > max_frame_id:
                max_frame_id = frame_id
    return max_frame_id + 1


def is_ref_sequence(clip_filename):
    """Check if a tar file is a ref (onboarding) sequence rather than a regular clip."""
    return clip_filename.startswith("obj_")


def load_mask_from_tar(tar, tar_names_set, prefix, frame_key, stream_id):
    """Load a pre-computed mask PNG from the tar file.

    Args:
        tar: Open tar file.
        tar_names_set: Set of tar member names (for O(1) lookup).
        prefix: Directory prefix inside tar (e.g. "obj_000001_up/" or "").
        frame_key: Frame key (e.g. "000000").
        stream_id: Camera stream id (e.g. "214-1").

    Returns a PIL Image in 'L' mode, or None if the mask file doesn't exist.
    """
    mask_filename = f"{prefix}{frame_key}.mask_{stream_id}.png"
    if mask_filename in tar_names_set:
        mask_file = tar.extractfile(mask_filename)
        mask = Image.open(mask_file).convert("L")
        return mask
    return None


def process_clip(clip, clips_input_dir, scenes_output_dir, args):
    # get clip id
    is_ref = is_ref_sequence(clip)
    if is_ref:
        # ref tars: "obj_000001_up.tar" -> "obj_000001_up"
        clip_name = clip.split(".tar")[0]
    else:
        # regular clips: "clip-XXXXXX.tar" -> "XXXXXX"
        clip_name = clip.split(".")[0].split("-")[1]

    # make scene folder and files for the scene
    scene_output_dir = os.path.join(scenes_output_dir, clip_name)

    # skip already converted scenes (check that all final JSON files exist)
    expected_jsons = [
        f"scene_camera_{s}.json" for s in args.camera_streams_names
    ] + [
        f"scene_gt_{s}.json" for s in args.camera_streams_names
    ] + [
        f"scene_gt_info_{s}.json" for s in args.camera_streams_names
    ]
    if os.path.isdir(scene_output_dir) and all(
        os.path.isfile(os.path.join(scene_output_dir, j)) for j in expected_jsons
    ):
        return

    # extract clip
    tar = tarfile.open(os.path.join(clips_input_dir, clip), "r")

    # detect directory prefix inside tar (ref static tars have e.g. "obj_000001_up/" prefix)
    tar_prefix = detect_tar_prefix(tar)
    tar_names_set = set(tar.getnames())

    # Check if any mask PNGs exist in the tar (ref static tars have them, dynamic don't)
    has_masks = any(name.endswith('.png') and '.mask_' in name for name in tar_names_set)

    os.makedirs(scene_output_dir, exist_ok=True)

    # make path of folders and folders
    # eg: STREAM_NAME, mask_STREAM_NAME, mask_visib_STREAM_NAME
    # also create path for each json file
    # eg: scene_camera_STREAM_NAME.json, scene_gt_STREAM_NAME.json, scene_gt_info_STREAM_NAME.json
    # create a dictionary for all camera streams
    clip_stream_paths = {}
    for stream_name in args.camera_streams_names:
        # directories
        stream_image_dir = os.path.join(scene_output_dir, stream_name)
        os.makedirs(stream_image_dir, exist_ok=True)
        clip_stream_paths[stream_name] = stream_image_dir
        if has_masks or not is_ref:
            stream_mask_dir = os.path.join(scene_output_dir, f"mask_{stream_name}")
            os.makedirs(stream_mask_dir, exist_ok=True)
            clip_stream_paths[f"mask_{stream_name}"] = stream_mask_dir
            stream_mask_visib_dir = os.path.join(
                scene_output_dir, f"mask_visib_{stream_name}"
            )
            os.makedirs(stream_mask_visib_dir, exist_ok=True)
            clip_stream_paths[f"mask_visib_{stream_name}"] = stream_mask_visib_dir
        # json files
        stream_scene_camera_json_path = os.path.join(
            scene_output_dir, f"scene_camera_{stream_name}.json"
        )
        clip_stream_paths[f"scene_camera_{stream_name}"] = stream_scene_camera_json_path
        stream_scene_gt_json_path = os.path.join(
            scene_output_dir, f"scene_gt_{stream_name}.json"
        )
        clip_stream_paths[f"scene_gt_{stream_name}"] = stream_scene_gt_json_path
        stream_scene_gt_info_json_path = os.path.join(
            scene_output_dir, f"scene_gt_info_{stream_name}.json"
        )
        clip_stream_paths[f"scene_gt_info_{stream_name}"] = (
            stream_scene_gt_info_json_path
        )

    # make a dict of dicts with stream name as keys
    scene_camera_data = {}
    scene_gt_data = {}
    scene_gt_info_data = {}
    for stream_name in args.camera_streams_names:
        # add an empty dict indicating the stream name
        scene_camera_data[stream_name] = {}
        scene_gt_data[stream_name] = {}
        scene_gt_info_data[stream_name] = {}

    # loop over all frames
    num_frames = get_number_of_frames_prefixed(tar, tar_prefix)
    for frame_id in range(num_frames):
        frame_key = f"{frame_id:06d}"
        # frame_key_prefixed is used for clip_util calls and direct tar access
        frame_key_prefixed = f"{tar_prefix}{frame_key}"

        # Load camera parameters.
        # from FRAME_ID.cameras.json
        frame_camera, _ = clip_util.load_cameras(tar, frame_key_prefixed)
        ## read FRAME_ID.objects.json
        frame_objects = clip_util.load_object_annotations(tar, frame_key_prefixed)
        if frame_objects is None:
            frame_objects = {}

        # read calibration json as it is
        camera_json_file_name = f"{frame_key_prefixed}.cameras.json"
        camera_json_file = tar.extractfile(camera_json_file_name)
        frame_camera_data = json.load(camera_json_file)

        # read FRAME_ID.info.json
        frame_info_file_name = f"{frame_key_prefixed}.info.json"
        frame_info_file = tar.extractfile(frame_info_file_name)
        frame_info_data = json.load(frame_info_file)

        # loop over all camera streams
        for stream_index, stream_name in enumerate(args.camera_streams_names):
            stream_id = args.camera_streams_id[stream_index]

            # load the image corresponding to the stream and frame
            image = clip_util.load_image(tar, frame_key_prefixed, stream_id)
            # if image is rgb (3 channels), convert to BGR
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # save the image
            image_path = os.path.join(
                clip_stream_paths[stream_name], frame_key + ".jpg"
            )
            cv2.imwrite(image_path, image)

            # filling scene_camera.json

            # get T_world_from_camera
            T_world_from_camera = frame_camera[stream_id].T_world_from_eye

            T_world_to_camera = np.linalg.inv(T_world_from_camera)

            # get camera parameters
            calibration = frame_camera_data[stream_id]["calibration"]

            # add frame scene_camera data
            scene_camera_data[stream_name][int(frame_id)] = {
                "cam_model": calibration,
                "device": frame_info_data["device"],
                "image_timestamps_ns": frame_info_data["image_timestamps_ns"][
                    stream_id
                ],
                # "cam_K":  # not used as cam_model exists
                # "depth_scale":  # also not used
                # convert translation from meter to mm
                "cam_R_w2c": T_world_to_camera[:3, :3].flatten().tolist(),
                "cam_t_w2c": (T_world_to_camera[:3, 3] * 1000).tolist(),
            }

            # Camera parameters of the current image.
            # camera_model = frame_camera[stream_id]

            frame_scene_gt_data = []
            frame_scene_gt_info_data = []
            # loop with enumerate over all objects in the frame
            for anno_id, obj_key in enumerate(frame_objects):
                obj_data = frame_objects[obj_key][0]

                width = frame_camera_data[stream_id]["calibration"]["image_width"]
                height = frame_camera_data[stream_id]["calibration"]["image_height"]

                # Check if the object is visible in the current stream
                if stream_id not in obj_data.get("visibilities_modeled", {}):
                    # Object not visible in this stream — dummy annotation
                    object_frame_scene_gt_anno = {
                        "obj_id": int(obj_key),
                        "cam_R_m2c": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                        "cam_t_m2c": [-1, -1, -1],
                    }
                    object_frame_scene_gt_info_anno = {
                        "bbox_obj": [-1, -1, -1, -1],
                        "bbox_visib": [-1, -1, -1, -1],
                        "px_count_all": 0,
                        "px_count_visib": 0,
                        "visib_fract": 0,
                    }
                    if has_masks or not is_ref:
                        mask = Image.new("L", (width, height), 0)
                        mask_visib = Image.new("L", (width, height), 0)
                    else:
                        mask = None
                        mask_visib = None

                elif is_ref:
                    # --- Ref sequence: masks from separate PNG files, no RLE in objects.json ---

                    # Transformation from the model to the world space.
                    T_world_from_model = clip_util.se3_from_dict(
                        obj_data["T_world_from_object"]
                    )
                    T_camera_from_model = (
                        np.linalg.inv(T_world_from_camera) @ T_world_from_model
                    )

                    object_frame_scene_gt_anno = {
                        "obj_id": int(obj_key),
                        "cam_R_m2c": T_camera_from_model[:3, :3].flatten().tolist(),
                        "cam_t_m2c": (T_camera_from_model[:3, 3] * 1000).tolist(),
                    }

                    # Load mask from separate PNG file in the tar (if available for this stream)
                    tar_mask = load_mask_from_tar(tar, tar_names_set, tar_prefix, frame_key, stream_id)
                    if tar_mask is not None:
                        mask = tar_mask
                        # For ref sequences, use the same mask as mask_visib
                        # (ref sequences show objects without heavy occlusion)
                        mask_visib = tar_mask.copy()
                    else:
                        # No mask file for this stream — skip mask output
                        mask = None
                        mask_visib = None

                    visib_fract = obj_data["visibilities_modeled"].get(stream_id, 0.0)

                    if mask is not None:
                        px_count_all = cv2.countNonZero(np.array(mask))
                        px_count_visib = cv2.countNonZero(np.array(mask_visib))

                        bbox_obj = obj_data["boxes_amodal"].get(stream_id)
                        if bbox_obj is not None:
                            # convert from xyxy to xywh
                            bbox_obj = [
                                int(bbox_obj[0]),
                                int(bbox_obj[1]),
                                int(bbox_obj[2] - bbox_obj[0]),
                                int(bbox_obj[3] - bbox_obj[1]),
                            ]
                        elif px_count_all > 0:
                            ys, xs = np.asarray(mask).nonzero()
                            bbox_obj = misc.calc_2d_bbox(xs, ys, mask.size)
                            bbox_obj = [int(x) for x in bbox_obj]
                        else:
                            bbox_obj = [-1, -1, -1, -1]

                        if px_count_visib > 0:
                            ys, xs = np.asarray(mask_visib).nonzero()
                            bbox_visib = misc.calc_2d_bbox(xs, ys, mask_visib.size)
                            bbox_visib = [int(x) for x in bbox_visib]
                        else:
                            bbox_visib = [-1, -1, -1, -1]
                    else:
                        px_count_all = -1
                        px_count_visib = -1
                        bbox_obj = obj_data["boxes_amodal"].get(stream_id)
                        if bbox_obj is not None:
                            bbox_obj = [
                                int(bbox_obj[0]),
                                int(bbox_obj[1]),
                                int(bbox_obj[2] - bbox_obj[0]),
                                int(bbox_obj[3] - bbox_obj[1]),
                            ]
                        else:
                            bbox_obj = [-1, -1, -1, -1]
                        bbox_visib = [-1, -1, -1, -1]

                    object_frame_scene_gt_info_anno = {
                        "bbox_obj": bbox_obj,
                        "bbox_visib": bbox_visib,
                        "px_count_all": px_count_all,
                        "px_count_visib": px_count_visib,
                        "visib_fract": visib_fract,
                    }

                elif not obj_data["masks_amodal"][stream_id]["rle"]:
                    # Regular clip but RLE mask is empty — very low visibility object
                    object_frame_scene_gt_anno = {
                        "obj_id": int(obj_key),
                        "cam_R_m2c": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                        "cam_t_m2c": [-1, -1, -1],
                    }
                    object_frame_scene_gt_info_anno = {
                        "bbox_obj": [-1, -1, -1, -1],
                        "bbox_visib": [-1, -1, -1, -1],
                        "px_count_all": 0,
                        "px_count_visib": 0,
                        "visib_fract": 0,
                    }
                    mask = Image.new("L", (width, height), 0)
                    mask_visib = Image.new("L", (width, height), 0)

                else:
                    # --- Regular clip: masks from RLE in objects.json ---

                    # Transformation from the model to the world space.
                    T_world_from_model = clip_util.se3_from_dict(
                        obj_data["T_world_from_object"]
                    )
                    T_camera_from_model = (
                        np.linalg.inv(T_world_from_camera) @ T_world_from_model
                    )

                    object_frame_scene_gt_anno = {
                        "obj_id": int(obj_key),
                        "cam_R_m2c": T_camera_from_model[:3, :3].flatten().tolist(),
                        "cam_t_m2c": (T_camera_from_model[:3, 3] * 1000).tolist(),
                    }

                    # read amodal masks
                    rle_dict = obj_data["masks_amodal"][stream_id]
                    if not rle_dict["rle"]:
                        print(
                            "RLE mask is empty!",
                            "For scene_id:{}, frame_id: {}, obj_id: {}.".format(
                                clip_name, frame_id, obj_key
                            ),
                            "This case shouldn't happen. Maybe that is an edge case That is not covered here.",
                            "The process will exit.",
                        )
                        exit()
                    else:
                        mask = custom_rle_to_mask(
                            rle_dict["height"], rle_dict["width"], rle_dict["rle"]
                        )
                        mask = Image.fromarray(mask * 255)
                        mask = mask.convert("L")

                    # read modal mask
                    rle_dict = obj_data["masks_modal"][stream_id]
                    if not rle_dict["rle"]:
                        mask_visib = Image.new(
                            "L", (rle_dict["width"], rle_dict["height"]), 0
                        )
                    else:
                        mask_visib = custom_rle_to_mask(
                            rle_dict["height"], rle_dict["width"], rle_dict["rle"]
                        )
                        mask_visib = Image.fromarray(mask_visib * 255)
                        mask_visib = mask_visib.convert("L")

                    px_count_all = cv2.countNonZero(np.array(mask))
                    px_count_visib = cv2.countNonZero(np.array(mask_visib))
                    visibilities_modeled = obj_data["visibilities_modeled"][stream_id]
                    visibilities_predicted = obj_data["visibilities_predicted"][
                        stream_id
                    ]
                    visib_fract = min(visibilities_modeled, visibilities_predicted)

                    bbox_obj = obj_data["boxes_amodal"][stream_id]
                    # change bbox from xyxy to xywh
                    bbox_obj = [
                        bbox_obj[0],
                        bbox_obj[1],
                        bbox_obj[2] - bbox_obj[0],
                        bbox_obj[3] - bbox_obj[1],
                    ]
                    bbox_obj = [int(val) for val in bbox_obj]
                    if px_count_visib > 0:
                        ys, xs = np.asarray(mask_visib).nonzero()
                        im_size = mask_visib.size
                        bbox_visib = misc.calc_2d_bbox(xs, ys, im_size)
                        bbox_visib = [int(x) for x in bbox_visib]
                    else:
                        bbox_visib = [-1, -1, -1, -1]
                    object_frame_scene_gt_info_anno = {
                        "bbox_obj": bbox_obj,
                        "bbox_visib": bbox_visib,
                        "px_count_all": px_count_all,
                        "px_count_visib": px_count_visib,
                        "visib_fract": visib_fract,
                    }

                anno_id = f"{anno_id:06d}"
                # save mask and mask_visib (skip if mask is None, e.g. dynamic ref without mask PNGs)
                if mask is not None:
                    mask_path = os.path.join(
                        clip_stream_paths[f"mask_{stream_name}"],
                        frame_key + "_" + anno_id + ".png",
                    )
                    mask.save(mask_path)
                    mask_visib_path = os.path.join(
                        clip_stream_paths[f"mask_visib_{stream_name}"],
                        frame_key + "_" + anno_id + ".png",
                    )
                    mask_visib.save(mask_visib_path)

                frame_scene_gt_data.append(object_frame_scene_gt_anno)
                frame_scene_gt_info_data.append(object_frame_scene_gt_info_anno)

            scene_gt_data[stream_name][int(frame_id)] = frame_scene_gt_data
            scene_gt_info_data[stream_name][int(frame_id)] = frame_scene_gt_info_data

    # save scene_gt.json, scene_gt_info.json, scene_camera.json for each camera stream
    for stream_name in args.camera_streams_names:
        with open(clip_stream_paths[f"scene_camera_{stream_name}"], "w") as f:
            json.dump(scene_camera_data[stream_name], f, indent=4)
        with open(clip_stream_paths[f"scene_gt_{stream_name}"], "w") as f:
            json.dump(scene_gt_data[stream_name], f, indent=4)
        with open(clip_stream_paths[f"scene_gt_info_{stream_name}"], "w") as f:
            json.dump(scene_gt_info_data[stream_name], f, indent=4)


def custom_rle_to_mask(height, width, rle):
    """
    Convert custom RLE (Run-Length Encoding) to a binary mask using vectorized operations.

    Parameters:
    - height (int): The height of the mask.
    - width (int): The width of the mask.
    - rle (list): The custom RLE list [start, length, start, length, ...].

    Returns:
    - np.ndarray: The binary mask.
    """
    # Create an empty mask
    mask = np.zeros(height * width, dtype=np.uint8)

    # Convert RLE pairs into start and end indices
    starts = np.array(rle[0::2])
    lengths = np.array(rle[1::2])
    ends = starts + lengths

    # Create an array of indices corresponding to the runs
    run_lengths = np.concatenate(
        [np.arange(start, end) for start, end in zip(starts, ends)]
    )

    # Set those indices in the mask to 1
    mask[run_lengths] = 1

    # Reshape the flat array into a 2D mask
    return mask.reshape((height, width))


if __name__ == "__main__":
    main()
