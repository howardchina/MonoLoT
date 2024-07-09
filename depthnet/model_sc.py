from .networks import *
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
import os
from .networks.layers import *
import random

EPS = 1e-7


class EstimateDepthSC():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.height = cfgs.get('height', 256)
        self.width = cfgs.get('width', 320)
        self.batch_size = cfgs.get('batch_size', 64)
        
        # checking height and width are multiples of 32
        assert self.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.width % 32 == 0, "'width' must be a multiple of 32"
        
        self.device = cfgs.get('device', 'cpu')
        self.scales = cfgs.get('scales', [0,1,2,3])
        self.num_scales = len(self.scales)
        self.frame_ids = cfgs.get('frame_ids', [0,-1,1])
        self.num_pose_frames = 2
        self.disable_automasking = cfgs.get('disable_automasking', False)

        assert self.frame_ids[0] == 0, "frame_ids must start with 0"

        # depth
        self.min_depth = cfgs.get('min_depth', '0.1')
        self.max_depth = cfgs.get('max_depth', '100.0')
        self.min_gt_depth = cfgs.get('min_gt_depth', '0.001')
        self.max_gt_depth = cfgs.get('max_gt_depth', '1.')
        
        self.model_str = cfgs.get('model_str', 'monodepth2')
        self.num_layers = cfgs.get('num_layers', 18)
        self.weights_init = cfgs.get('weights_init', "pretrained")
        if self.model_str == "monodepth2":
            self.net_depth_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained")
            self.net_depth_decoder = DepthDecoder(self.net_depth_encoder.num_ch_enc, self.scales,)
            # pose
            self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained", num_input_images=self.num_pose_frames)
            self.net_pose_decoder = PoseDecoder(self.net_pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        elif self.model_str in ["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"]:
            self.drop_path = cfgs.get('drop_path', 0.2)
            self.net_depth_encoder = LiteMono(model=self.model_str, drop_path_rate=self.drop_path, width=self.width, height=self.height)
            self.net_depth_decoder = DepthDecoderV2(self.net_depth_encoder.num_ch_enc, self.scales)
            # pose
            self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained", num_input_images=self.num_pose_frames)
            self.net_pose_decoder = PoseDecoderV2(self.net_pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        elif self.model_str == 'monovit':
            self.net_depth_encoder = mpvit_small()
            self.net_depth_decoder = MonovitDecoder()
            # pose
            self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained", num_input_images=self.num_pose_frames)
            self.net_pose_decoder = PoseDecoder(self.net_pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        else:
            raise NotImplementedError


        # optim
        self.start_epoch = cfgs.get('start_epoch', 0)
        self.num_epochs = cfgs.get('num_epochs', 20)
        self.lr = cfgs.get('lr', [0.0001, 5e-6, 36, 0.0001, 1e-5, 36])
        self.weight_decay = cfgs.get('weight_decay', 0.02)
        self.disparity_smoothness = cfgs.get('disparity_smoothness', 0.001)
        
        # matcher loss
        self.disable_matcher = cfgs.get('disable_matcher', False)
        self.confidence = cfgs.get('confidence', 0.9)
        self.matcher_loss_alpha = cfgs.get('matcher_loss_alpha', 0.2)
        self.half_epoch_matcher = cfgs.get('half_epoch_matcher', False)
        self.matcher_loss_delta = cfgs.get('matcher_loss_delta', 0)
        
        self.network_names = [k for k in vars(self) if k.startswith('net')]
        # optimizer
        if self.model_str == 'monovit' or 'lite-mono':
            self.make_optimizer = lambda optim_dict: torch.optim.AdamW(
                optim_dict["parameters"], lr=optim_dict["lr"], weight_decay=self.weight_decay)
        elif self.model_str == "monodepth2":
            self.make_optimizer = lambda optim_dict: torch.optim.Adam(
                optim_dict["parameters"], lr=optim_dict["lr"])
        
        # ratio consistency
        self.ratio_consistency = cfgs.get('ratio_consistency', False)
        self.ratio_consistency_crop = cfgs.get('ratio_consistency_crop', False)
        self.ratio_consistency_normalization = cfgs.get('ratio_consistency_normalization', False)
        self.ratio_consistency_scales_normalization = cfgs.get('ratio_consistency_scales_normalization', False)
        self.weight_ratio_consistency_crop = cfgs.get('weight_ratio_consistency_crop', 1.0)
        self.align_crop_position = cfgs.get('align_crop_position', False)
        
        # geometry loss
        self.geometry_loss = cfgs.get('geometry_loss', False)
        self.geometry_loss_disp_mode = cfgs.get('geometry_loss_disp_mode', False)

        # load
        self.load_weights_folder = cfgs.get('load_weights_folder', None)
        self.mypretrain = cfgs.get('mypretrain', None)
        self.not_load_nets = cfgs.get('not_load_nets', ())
        self.not_load_optimizer = cfgs.get('not_load_optimizer', ())
        self.models_to_load = cfgs.get('models_to_load', [])
        
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        
        # no grad layers
        self.ssim = SSIM()
        for scale in self.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)
            setattr(self, "backproject_depth_{}".format(scale), BackprojectDepth(self.batch_size, h, w))
            setattr(self, "project_3d_{}".format(scale), Project3D(self.batch_size, h, w))
            
        self.other_param_names = ['ssim']
        for scale in self.scales:
            self.other_param_names += ["backproject_depth_{}".format(scale), "project_3d_{}".format(scale)]

    def init_optimizers(self):
        # optim
        self.optimizer_names = []
        self.parameters_depth = []
        self.parameters_pose = []
        for net_name in self.network_names:
            if not any([p.requires_grad for p in getattr(self, net_name).parameters()]):
                continue
            if net_name.startswith('net_depth'):
                self.parameters_depth += list(getattr(self, net_name).parameters())
            elif net_name.startswith('net_pose'):
                self.parameters_pose += list(getattr(self, net_name).parameters())
        self.optimizer_depth = self.make_optimizer({"parameters": self.parameters_depth, "lr": self.lr[0]})
        self.optimizer_pose = self.make_optimizer({"parameters": self.parameters_pose, "lr": self.lr[3]})
        self.optimizer_names += ["optimizer_depth", "optimizer_pose"]
        
        # scheduler
        self.scheduler_names = []
        if self.model_str == 'lite-mono':
            self.scheduler_depth_lr = ChainedScheduler(
                self.optimizer_depth,
                T_0=int(self.lr[2]), T_mul=1, eta_min=self.lr[1], last_epoch=self.start_epoch-1,
                max_lr=self.lr[0], warmup_steps=0, gamma=0.9)
            self.scheduler_pose_lr = ChainedScheduler(
                self.optimizer_pose,
                T_0=int(self.lr[5]), T_mul=1, eta_min=self.lr[4], last_epoch=self.start_epoch-1,
                max_lr=self.lr[3], warmup_steps=0, gamma=0.9)
        elif self.model_str == 'monovit':
            self.scheduler_depth_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_depth, 0.9)
            self.scheduler_pose_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pose, 0.9)
        elif self.model_str == 'monodepth2':
            self.scheduler_depth_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_depth, self.num_epochs - 5, 0.1)
            self.scheduler_pose_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_pose, self.num_epochs - 5, 0.1)

        self.scheduler_names += ["scheduler_depth_lr", "scheduler_pose_lr"]

    def load_model(self):
        """Load model(s) from disk
        """
        self.load_weights_folder = os.path.expanduser(self.load_weights_folder)

        assert os.path.isdir(self.load_weights_folder), \
            "Cannot find folder {}".format(self.load_weights_folder)
        print("loading model from folder {}".format(self.load_weights_folder))

        for n in self.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.load_weights_folder, "{}.pth".format(n))

            model_dict = getattr(self, n).state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            getattr(self, n).load_state_dict(model_dict)

        # loading optimizer state
        optimizer_depth_load_path = os.path.join(self.load_weights_folder, "optimizer_depth.pth")
        optimizer_pose_load_path = os.path.join(self.load_weights_folder, "optimizer_pose.pth")
        if os.path.isfile(optimizer_depth_load_path) and os.path.isfile(optimizer_pose_load_path):
            print("Loading optimizer weights")
            self.optimizer_depth.load_state_dict(torch.load(optimizer_depth_load_path))
            self.optimizer_pose.load_state_dict(torch.load(optimizer_pose_load_path))
        else:
            print("Cannot find optimizer weights so Adam is randomly initialized")

    def load_pretrain(self):
        # only designed for lite-mono
        self.mypretrain = os.path.expanduser(self.mypretrain)
        path = self.mypretrain
        model_dict = self.net_depth_encoder.state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.net_depth_encoder.load_state_dict(model_dict)
        print('mypretrain loaded.')
    
    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names and k not in self.not_load_nets:
                print("Loading ", k)
                model_dict = getattr(self, k).state_dict()
                getattr(self, k).load_state_dict({k: v for k, v in cp[k].items() if k in model_dict})

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names and k not in self.not_load_optimizer:
                print("Loading ", k)
                getattr(self, k).load_state_dict(cp[k])
                
    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
            
        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def backward(self, losses):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        losses["loss"].backward()
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()

    def forward(self, inputs):
        """Feedforward once."""
        for key, ipt in inputs.items():
            if "correspondences" in key:
                inputs[key] = ipt
            else:
                inputs[key] = ipt.to(self.device)
        
        # we only feed the image with frame_id 0 through the depth encoder
        features = self.net_depth_encoder(inputs["color_aug", 0, 0])
        outputs = self.net_depth_decoder(features)
        outputs.update(self.predict_poses(inputs))
        

        
        # shuffle
        outputs["do_shuffle"] = random.random() > 0.5
        if self.ratio_consistency and outputs["do_shuffle"]:
            direction = random.random() > 0.5
            inputs["shuffle_color_aug", 0, 0] = self.layer_shuffle(inputs["color_aug", 0, 0], direction)
            shuffle_features = self.net_depth_encoder(inputs["shuffle_color_aug", 0, 0])
            shuffle_outputs = self.net_depth_decoder(shuffle_features)
            for scale in self.scales:
                outputs[("shuffle_disp", scale)] = self.layer_shuffle(shuffle_outputs[("disp", scale)], direction)
        elif self.ratio_consistency_crop and outputs["do_shuffle"]:
            b, _, h, w = inputs["color_aug", 0, 0].shape
            crop_info = self.get_crop_info(h // 2**(self.num_scales-1), w // 2**(self.num_scales-1), align_crop_position=self.align_crop_position) * 2**(self.num_scales-1)
            inputs['crop_info'] = crop_info

            inputs["shuffle_color_aug", 0, 0] = self.layer_crop_shuffle(inputs["color_aug", 0, 0], crop_info)
            shuffle_features = self.net_depth_encoder(inputs["shuffle_color_aug", 0, 0])
            shuffle_outputs = self.net_depth_decoder(shuffle_features)
            for scale in self.scales:
                outputs[("shuffle_disp", scale)] = self.layer_crop_shuffle(shuffle_outputs[("disp", scale)], crop_info // 2**scale)
        
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses
    
    def layer_shuffle(self, input_raw, direction=True):
        # input raw: (b, 3, h, w)
        if direction:
            chunk_0, chunk_1 = torch.chunk(input_raw, 2)
            chunk_0_up, chunk_0_botton = torch.chunk(chunk_0, 2, 2)
            chunk_1_up, chunk_1_botton = torch.chunk(chunk_1, 2, 2)
            chunk_0_up_1_botton = torch.cat([chunk_0_up, chunk_1_botton], 2)
            chunk_1_up_0_botton = torch.cat([chunk_1_up, chunk_0_botton], 2)
            shuffle_input = torch.cat([chunk_0_up_1_botton, chunk_1_up_0_botton], 0)
        else:
            chunk_0, chunk_1 = torch.chunk(input_raw, 2)
            chunk_0_left, chunk_0_right = torch.chunk(chunk_0, 2, 3)
            chunk_1_left, chunk_1_right = torch.chunk(chunk_1, 2, 3)
            chunk_0_left_1_right = torch.cat([chunk_0_left, chunk_1_right], 3)
            chunk_1_left_0_right = torch.cat([chunk_1_left, chunk_0_right], 3)
            shuffle_input = torch.cat([chunk_0_left_1_right, chunk_1_left_0_right], 0)
        return shuffle_input
    
    def layer_crop_shuffle(self, input_raw, crop_info):
        chunk_0, chunk_1 = torch.chunk(input_raw, 2)
        x1, y1, x1p, y1p, patch_width, patch_height, w, h = crop_info # [232 136  64   8  56  80 320 256] 
        
        split_xp = [x1p, patch_width, w - (x1p + patch_width)]
        split_yp = [y1p, patch_height, h - (y1p + patch_height)]
        
        chunk_0_left_middle_right = torch.split(chunk_0, split_xp, dim=3)
        chunk_0_middle_up_middle_center_middle_right = torch.split(chunk_0_left_middle_right[1], split_yp, dim=2)
        
        split_x = [x1, patch_width, w - (x1 + patch_width)]
        split_y = [y1, patch_height, h - (y1 + patch_height)]
        
        chunk_1_left_middle_right = torch.split(chunk_1, split_x, dim=3)
        chunk_1_middle_up_middle_center_middle_right = torch.split(chunk_1_left_middle_right[1], split_y, dim=2)
        
        shuffle_0_middle = torch.cat([chunk_0_middle_up_middle_center_middle_right[0], chunk_1_middle_up_middle_center_middle_right[1], chunk_0_middle_up_middle_center_middle_right[2]], 2)
        shuffle_0 = torch.cat([chunk_0_left_middle_right[0], shuffle_0_middle, chunk_0_left_middle_right[2]], 3)
        
        shuffle_1_middle = torch.cat([chunk_1_middle_up_middle_center_middle_right[0], chunk_0_middle_up_middle_center_middle_right[1], chunk_1_middle_up_middle_center_middle_right[2]], 2)
        shuffle_1 = torch.cat([chunk_1_left_middle_right[0], shuffle_1_middle, chunk_1_left_middle_right[2]], 3)
        
        shuffle_input = torch.cat([shuffle_0, shuffle_1], 0)
        return shuffle_input

    def get_crop_info(self, h, w, min_patch_ratio=0.6, max_path_ratio=0.8, align_crop_position=False):
        min_width = round(min_patch_ratio * w)
        max_width = round(max_path_ratio * w)
        
        min_height = round(min_patch_ratio * h)
        max_height = round(max_path_ratio * h)
        
        patch_width = np.random.randint(min_width, max_width+1)
        patch_height = np.random.randint(min_height, max_height+1)
        
        x1 = np.random.randint(0, w - patch_width)
        y1 = np.random.randint(0, h - patch_height)
        # x2 = x1 + patch_width
        # y2 = y1 + patch_height
        
        if align_crop_position:
            x1p = x1
            y1p = y1
        else:
            x1p = np.random.randint(0, w - patch_width)
            y1p = np.random.randint(0, h - patch_height)
        # x2p = x1p + patch_width
        # y2p = y1p + patch_height

        return np.array([x1, y1, x1p, y1p, patch_width, patch_height, w, h], dtype=int)

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

            for f_i in self.frame_ids[1:]:
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.net_pose_encoder(torch.cat(pose_inputs, 1))]
                axisangle, translation = self.net_pose_decoder(pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        else:
            raise NotImplementedError

        return outputs
    
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.height, self.width], mode="bilinear", align_corners=False)
            source_scale = 0
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]
                _backproject_depth = getattr(self, "backproject_depth_{}".format(source_scale))
                cam_points = _backproject_depth(
                    depth, inputs[("inv_K", source_scale)])
                _project_3d = getattr(self, "project_3d_{}".format(source_scale))
                
                pix_coords = _project_3d(
                        cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="zeros", align_corners=True)

                outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]
    
    
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            valid_masks = []
            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                valid_masks.append((pred.abs().mean(1, True) > 1e-3).float())

            reprojection_losses = torch.cat(reprojection_losses, 1)
            valid_masks = torch.cat(valid_masks + valid_masks, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    # if camera does not move, pred and target are the same, so that loss=0
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            # [not move, corretlymatch]
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, idxs = torch.min(combined, dim=1, keepdim=True)
            valid_mask = torch.gather(valid_masks, 1, idxs)

            # true means corretly match, false means not move
            outputs["identity_selection/{}".format(scale)] = torch.squeeze(
                idxs > identity_reprojection_loss.shape[1] - 1, dim=1).float()

            loss += self.mean_on_mask(to_optimise, valid_mask)
        
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/photometric_{}".format(scale)] = loss

        if not self.disable_matcher:
            matcher_loss = 0
            for frame_id in self.frame_ids[1:]:
                # correspondences = self.matcher(inputs[('color', 0, source_scale)], inputs[('color', frame_id, source_scale)])
                correspondences = inputs[('correspondences', 0, frame_id)]
                for scale in self.scales:
                    matcher_loss += compute_matcher_errors_from_correspondences(correspondences, outputs[("sample", frame_id, scale)],
                                                                                self.width, self.height, self.batch_size, self.device,
                                                                                confidence=self.confidence, delta=self.matcher_loss_delta)
            matcher_loss *= self.matcher_loss_alpha
            total_loss += matcher_loss
            losses["loss/matcher"] = matcher_loss
            
        if self.ratio_consistency and outputs["do_shuffle"]:
            ratio_consistency_loss = 0
            for scale in self.scales:
                # ratio_consistency_loss += torch.abs(outputs[("shuffle_disp", scale)] - outputs[("disp", scale)]).mean()
                ratio_consistency_loss += self.compute_batch_image_shuffle_loss(outputs[("shuffle_disp", scale)], outputs[("disp", scale)], norm=self.ratio_consistency_normalization)
            total_loss += ratio_consistency_loss
            losses["loss/ratio_consistency"] = ratio_consistency_loss
        elif self.ratio_consistency_crop and outputs["do_shuffle"]:
            ratio_consistency_crop_loss = 0
            for scale in self.scales:
                ratio_consistency_crop_loss_tmp = torch.abs(outputs[("shuffle_disp", scale)] - outputs[("disp", scale)])
                if self.ratio_consistency_normalization:
                    ratio_consistency_crop_loss_tmp /= (outputs[("shuffle_disp", scale)] + outputs[("disp", scale)])
                if self.ratio_consistency_scales_normalization:
                    ratio_consistency_crop_loss_tmp /= (2 ** scale)
                ratio_consistency_crop_loss += self.compute_random_batch_image_shuffle_loss(ratio_consistency_crop_loss_tmp, inputs["crop_info"] // 2**scale).mean()
            ratio_consistency_crop_loss *= self.weight_ratio_consistency_crop
            total_loss += ratio_consistency_crop_loss
            losses["loss/ratio_consistency_crop"] = ratio_consistency_crop_loss

        total_loss /= len(self.scales)
        losses["loss"] = total_loss
        return losses

    def mean_on_mask(self, diff, valid_mask):
        mask = valid_mask.expand_as(diff)
        if mask.sum() > 100:
            mean_value = (diff * mask).sum() / mask.sum()
        else:
            mean_value = torch.tensor(0).float().to(self.device)
        return mean_value

    def compute_random_batch_image_shuffle_loss(self, l1, crop_info):
        b, _, h, w = l1.shape
        x1, y1, x1p, y1p, patch_width, patch_height, _, _ = crop_info
        mask = torch.ones_like(l1)
        min_x1p = max(x1p - 1, 0)
        max_x1p = min(x1p + patch_width + 1, w)
        min_y1p = max(y1p - 1, 0)
        max_y1p = min(y1p + patch_height + 1, h)
        mask[:b//2, :, min_y1p:min_y1p+2, min_x1p:max_x1p] = 0 # top
        mask[:b//2, :, max_y1p-2:max_y1p, min_x1p:max_x1p] = 0 # botton
        mask[:b//2, :, min_y1p:max_y1p, min_x1p:min_x1p+2] = 0 # left
        mask[:b//2, :, min_y1p:max_y1p, max_x1p-2:max_x1p] = 0 # right

        min_x1 = max(x1 - 1, 0)
        max_x1 = min(x1 + patch_width + 1, w)
        min_y1 = max(y1 - 1, 0)
        max_y1 = min(y1 + patch_height + 1, h)
        mask[b//2:, :, min_y1:min_y1+2, min_x1:max_x1] = 0 # top
        mask[b//2:, :, max_y1-2:max_y1, min_x1:max_x1] = 0 # botton
        mask[b//2:, :, min_y1:max_y1, min_x1:min_x1+2] = 0 # left
        mask[b//2:, :, min_y1:max_y1, max_x1-2:max_x1] = 0 # right

        return l1 * mask
    
    def compute_batch_image_shuffle_loss(self, pred, target, norm=False):
        mask = torch.ones_like(pred)
        b, _, h, w = pred.shape
        mask[:, :, :, w//2-1:w//2+1] = 0
        mask[:, :, h//2-1:h//2+1, :] = 0
        if norm:
            return ((torch.abs(pred - target) / (pred + target)) * mask).mean()
        else:
            return torch.abs((pred - target) * mask).mean()
    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    @torch.no_grad()
    def compute_depth_losses(self, inputs, outputs):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_losses = {}
        depth_gt = inputs["depth_gt"]
        b, c, gt_h, gt_w = depth_gt.shape
        mask = depth_gt > 0
        
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = F.interpolate(
            depth_pred, [gt_h, gt_w], mode="bilinear", align_corners=False)
        depth_pred = depth_pred.detach()

        
        depth_gt_flatten = depth_gt.view(self.batch_size, -1)
        depth_pred_flatten = depth_pred.view(self.batch_size, -1)
        mask_flatten = mask.view(self.batch_size, -1)
        
        med_gt, _ = torch.masked_fill(depth_gt_flatten, ~mask_flatten, float("nan")).nanmedian(dim=1, keepdim=True)
        med_pred, _ = torch.masked_fill(depth_pred_flatten, ~mask_flatten, float("nan")).nanmedian(dim=1, keepdim=True)

        ratios = med_gt / med_pred
        avg = torch.mean(ratios)
        med = torch.median(ratios)
        std = torch.std(ratios/med)

        depth_losses['ratio/mean'] = np.array(avg.cpu())
        depth_losses['ratio/med'] = np.array(med.cpu())
        depth_losses['ratio/std'] = np.array(std.cpu())
        
        depth_pred *= ratios[..., None, None]
        
        depth_pred = depth_pred[mask]
        depth_gt = depth_gt[mask]

        depth_pred = torch.clamp(depth_pred, min=self.min_gt_depth, max=self.max_gt_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            depth_losses[metric] = np.array(depth_errors[i].cpu())
            
        return depth_losses
