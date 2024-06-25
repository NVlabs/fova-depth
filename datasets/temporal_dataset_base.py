from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data


class TemporalDatasetBase(data.Dataset):
    def __init__(
        self,
        num_forward_context,
        num_backward_context,
        mode="index_context",
        forward_index_context_step=None,
        backward_index_context_step=None,
        nominal_forward_context_distance=None,
        forward_look_ahead=None,
        nominal_backward_context_distance=None,
        backward_look_ahead=None,
        normalize_trans=True,
        start_ref_frames_remove=0,
        end_ref_frames_remove=0,
        distance_filter_threshold=None,
        gpu_transforms=None,
        reference_frame_stride=1,
    ):
        super().__init__()
        # frame: a dict: in nominal_distance_context mode must has a 'to_world' key, which is a 4x4 matrix
        # sequence: a dict that at minimum contains a 'frame_list' key , which is a list of frames sorted by time

        # Has two modes: index_context and nominal_distance_context

        # child class should define a self.sequence_dict with a dict of sequences
        assert mode in ["index_context", "nominal_distance_context"]
        self.mode = mode
        # used in either mode
        self.num_forward_context = num_forward_context
        self.num_backward_context = num_backward_context
        self.normalize_trans = normalize_trans
        self.reference_frame_stride = reference_frame_stride

        self.gpu_transforms = gpu_transforms

        # used in index_context mode
        self.forward_index_context_step = forward_index_context_step
        self.backward_index_context_step = backward_index_context_step

        # used in nominal_distance_mode
        self.nominal_forward_context_distance = nominal_forward_context_distance
        self.forward_look_ahead = forward_look_ahead
        self.nominal_backward_context_distance = nominal_backward_context_distance
        self.backward_look_ahead = backward_look_ahead

        self.start_ref_frames_remove = start_ref_frames_remove
        self.end_ref_frames_remove = end_ref_frames_remove

        self.distance_filter_threshold = distance_filter_threshold

    def setup(self):
        if self.distance_filter_threshold is not None:
            self.filter_no_motion(self.distance_filter_threshold)

        if self.mode == "nominal_distance_context":
            self.add_context_to_frame_dict_nominal_distance()
        elif self.mode == "index_context":
            self.add_context_to_frame_dict_index()
        self.setup_sample_list()

    def filter_no_motion(self, threshold):
        for k, v in self.sequence_dict.items():
            frame_list = v["frame_list"]
            filtered_frame_list = []

            last_pos = np.array([np.inf] * 3)
            for frame in frame_list:
                curr_pos = frame["to_world"][:3, 3]
                if np.linalg.norm(last_pos - curr_pos) > threshold:
                    filtered_frame_list.append(frame)
                    last_pos = curr_pos
            v["frame_list"] = filtered_frame_list

    def add_context_to_frame_dict_index(self):
        for k, v in self.sequence_dict.items():
            frame_list = v["frame_list"]
            N = len(frame_list)
            for frame_idx in range(0, N):
                temp = frame_idx + self.forward_index_context_step
                if temp < N:
                    frame_list[frame_idx]["forward_context"] = temp
                else:
                    frame_list[frame_idx]["forward_context"] = -1

            for frame_idx in range(0, N):
                temp = frame_idx - self.backward_index_context_step
                if temp >= 0:
                    frame_list[frame_idx]["backward_context"] = temp
                else:
                    frame_list[frame_idx]["backward_context"] = -1

    def add_context_to_frame_dict_nominal_distance(self):
        for k, v in self.sequence_dict.items():
            frame_list = v["frame_list"]
            N = len(frame_list)
            positions = np.stack([f["to_world"][:3, 3] for f in frame_list], axis=0)
            # make forward context
            for frame_idx in range(0, N - 1):
                ref_pos = positions[frame_idx : frame_idx + 1]
                possible_source_pos = positions[
                    frame_idx + 1 : min(frame_idx + 1 + self.forward_look_ahead, N)
                ]
                dist = np.linalg.norm(ref_pos - possible_source_pos, axis=1)
                offset = np.argmin(np.abs(dist - self.nominal_forward_context_distance))
                frame_list[frame_idx]["forward_context"] = frame_idx + 1 + offset
            frame_list[-1]["forward_context"] = -1

            # make backward context
            for frame_idx in range(1, N):
                ref_pos = positions[frame_idx : frame_idx + 1]
                possible_source_pos = positions[
                    max(frame_idx - self.backward_look_ahead, 0) : frame_idx
                ]
                dist = np.linalg.norm(ref_pos - possible_source_pos, axis=1)
                offset = np.argmin(np.abs(dist - self.nominal_backward_context_distance))
                frame_list[frame_idx]["backward_context"] = frame_idx - (dist.shape[0] - offset)
            frame_list[0]["backward_context"] = -1

    def setup_sample_list(self):
        sample_list = []  # each value is (key in sequence dict, list of frame_list indices)
        for sequence_key, seq in self.sequence_dict.items():
            total_frames = len(seq["frame_list"])
            for ref_frame_idx in range(
                self.start_ref_frames_remove,
                total_frames - self.end_ref_frames_remove,
                self.reference_frame_stride,
            ):
                frame_indices = [ref_frame_idx]

                # add forward source frames
                curr_frame_idx = ref_frame_idx
                for _ in range(self.num_forward_context):
                    curr_frame_idx = seq["frame_list"][curr_frame_idx]["forward_context"]
                    frame_indices.append(curr_frame_idx)

                # add forward source frames
                curr_frame_idx = ref_frame_idx
                for _ in range(self.num_backward_context):
                    curr_frame_idx = seq["frame_list"][curr_frame_idx]["backward_context"]
                    frame_indices.append(curr_frame_idx)

                if -1 not in frame_indices:
                    sample_list.append((sequence_key, frame_indices))

        self.sample_list = sample_list

    def load_frame(self, sequence_key, frame_idx, index_in_sample):
        raise NotImplementedError

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        seq_key, frame_indices = self.sample_list[idx]
        sample = defaultdict(list)
        for i, frame_idx in enumerate(frame_indices):
            frame = self.load_frame(seq_key, frame_idx, index_in_sample=i)
            for k, v in frame.items():
                sample[k].append(v)

        for k, v in sample.items():
            sample[k] = torch.utils.data.default_collate(v)

        if self.normalize_trans:
            all_trans = sample["to_world"][:, :3, 3]
            shift = torch.mean(all_trans, dim=0, keepdim=True)
            sample["to_world"][:, :3, 3] -= shift
            sample["shift"] = shift

        sample["idx"] = idx
        return sample

    def calculate_baselines(self):
        baselines = []
        for k, v in self.sequence_dict.items():
            frame_list = v["frame_list"]
            N = len(frame_list)
            for i in range(N - 1):
                pos0 = frame_list[i]["to_world"][:3, 3]
                pos1 = frame_list[i + 1]["to_world"][:3, 3]
                baseline = np.linalg.norm(pos0 - pos1)
                baselines.append(baseline)
        return baselines
