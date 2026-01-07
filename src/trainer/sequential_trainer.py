"""
Sequential Trainer for GAOT.
Handles time-dependent datasets with autoregressive prediction capabilities.
"""
import torch
import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm

from ..core.base_trainer import BaseTrainer
from ..core.trainer_utils import move_to_device, denormalize_data
from ..datasets.sequential_data_processor import SequentialDataProcessor
from ..datasets.graph_builder import GraphBuilder
from ..datasets.data_utils import TestDataset, collate_sequential_batch
from ..model.gaot import GAOT
from ..utils.metrics import compute_batch_errors, compute_final_metric
from ..utils.plotting import plot_estimates, create_sequential_animation


class SequentialTrainer(BaseTrainer):
    """
    Sequential trainer for sequential (time-dependent) problems.
    Automatically handles both fixed and variable coordinate modes.
    Supports autoregressive prediction and multiple stepper modes.
    """
    
    def __init__(self, config):
        # Initialize data processor
        self.data_processor = None
        self.graph_builder = None
        
        # Coordinate mode and data info
        self.coord_mode = None  # Will be determined from data
        self.coord_dim = None
        self.latent_tokens_coord = None
        self.coord = None       # For fx mode
        
        # Sequential-specific attributes
        self.stats = None
        self.max_time_diff = None
        self.stepper_mode = None
        self.t_values = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        super().__init__(config)
    
    def init_dataset(self, dataset_config):
        """Initialize dataset and data loaders for sequential data."""
        print("Initializing sequential dataset...")
        
        self.data_processor = SequentialDataProcessor(
            dataset_config=dataset_config,
            metadata=self.metadata,
            dtype=self.dtype
        )

        data_splits, is_variable_coords = self.data_processor.load_and_process_data()

        self.coord_mode = 'vx' if is_variable_coords else 'fx'
        print(f"Detected coordinate mode: {self.coord_mode}")
        
        self.max_time_diff = self.data_processor.max_time_diff
        self.time_step = self.data_processor.time_step
        self.stepper_mode = self.data_processor.stepper_mode
        self.t_values = self.data_processor.t_values
        self.stats = self.data_processor.stats
        
        latent_queries = self.data_processor.generate_latent_queries(
            self.model_config.latent_tokens_size
        )
        self.latent_tokens_coord = latent_queries
        
        coord_sample = (data_splits['train']['x'] if is_variable_coords 
                       else data_splits['train']['x'])
        self.coord_dim = coord_sample.shape[-1]
        
        u_sample = data_splits['train']['u']
        c_sample = data_splits['train']['c']
        self.num_output_channels = u_sample.shape[-1]
        
        # Compute input channels: u + time(2) + optional c + optional conditional_norm(-1)
        self.num_input_channels = u_sample.shape[-1] + 2  # u + start_time + time_diff
        if c_sample is not None:
            self.num_input_channels += c_sample.shape[-1]  # add c channels
        
        # Account for conditional normalization
        if getattr(self.model_config, 'use_conditional_norm', False):
            self.num_input_channels -= 1  # one less due to conditional norm
        
        if is_variable_coords:
            # Variable coordinates mode - need to build graphs
            self._init_variable_coords_mode(data_splits)
        else:
            # Fixed coordinates mode - simpler setup
            self._init_fixed_coords_mode(data_splits)

        print("Sequential dataset initialization complete.")
    
    def _init_variable_coords_mode(self, data_splits):
        """Initialize for variable coordinates mode."""
        print("Setting up variable coordinates mode for sequential data...")
        
        # Create graph builder
        neighbor_search_method = self.model_config.args.magno.neighbor_search_method
        self.graph_builder = GraphBuilder(neighbor_search_method=neighbor_search_method)
        
        # Get graph building parameters
        gno_radius = getattr(self.model_config.args.magno, 'radius', 0.033)
        scales = getattr(self.model_config.args.magno, 'scales', [1.0])
        
        # Build graphs for all splits
        all_graphs = self.graph_builder.build_all_graphs(
            data_splits=data_splits,
            latent_queries=self.latent_tokens_coord,
            gno_radius=gno_radius,
            scales=scales,
            build_train=self.setup_config.train
        )
        
        # Create data loaders with graphs
        loader_kwargs = {
            'encoder_graphs': {
                'train': all_graphs['train']['encoder'] if all_graphs['train'] else None,
                'val': all_graphs['val']['encoder'] if all_graphs['val'] else None,
                'test': all_graphs['test']['encoder']
            },
            'decoder_graphs': {
                'train': all_graphs['train']['decoder'] if all_graphs['train'] else None,
                'val': all_graphs['val']['decoder'] if all_graphs['val'] else None,
                'test': all_graphs['test']['decoder']
            }
        }
        
        loaders = self.data_processor.create_sequential_data_loaders(
            data_splits=data_splits,
            is_variable_coords=True,
            **loader_kwargs
        )
        
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']
    
    def _init_fixed_coords_mode(self, data_splits):
        """Initialize for fixed coordinates mode."""
        print("Setting up fixed coordinates mode for sequential data...")
        
        self.coord = self.data_processor.coord_scaler(data_splits['train']['x'])
        
        loaders = self.data_processor.create_sequential_data_loaders(
            data_splits=data_splits,
            is_variable_coords=False
        )
        
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']
    
    def init_model(self, model_config):
        """Initialize the GAOT model for sequential data."""
        model_config.args.magno.coord_dim = self.coord_dim
        
        self.model = GAOT(
            input_size=self.num_input_channels,
            output_size=self.num_output_channels,
            config=model_config
        )
        
        print(f"Initialized {model_config.name} model for sequential data with {self.coord_dim}D coordinates")
    
    def train_step(self, batch):
        """Perform one training step."""
        if self.coord_mode == 'fx':
            return self._train_step_fixed_coords(batch)
        else:
            return self._train_step_variable_coords(batch)
    
    def _train_step_fixed_coords(self, batch):
        """Training step for fixed coordinates mode."""
        x_batch, y_batch = batch
        
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        coord = self.coord.to(self.device)

        # Handle conditional normalization
        if getattr(self.model_config, 'use_conditional_norm', False):
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord,
                pndata=x_batch[..., :-1],         # exclude last time feature
                condition=x_batch[..., 0, -2:-1]  # condition from time features
            )
        else:
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord,
                pndata=x_batch
            )
        
        return self.loss_fn(pred, y_batch)
    
    def _train_step_variable_coords(self, batch):
        """Training step for variable coordinates mode."""
        if len(batch) == 3:
            x_batch, y_batch, coord_batch = batch
        else:
            x_batch, y_batch, coord_batch, encoder_graph_batch, decoder_graph_batch = batch
            encoder_graph_batch = move_to_device(encoder_graph_batch, self.device)
            decoder_graph_batch = move_to_device(decoder_graph_batch, self.device)
        
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        coord_batch = coord_batch.to(self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        # breakpoint()
        # Handle conditional normalization
        if getattr(self.model_config, 'use_conditional_norm', False):
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord_batch,
                pndata=x_batch[..., :-1],          # exclude last time feature
                condition=x_batch[..., 0, -2:-1],  # condition from time features
                encoder_nbrs=encoder_graph_batch if len(batch) > 3 else None,
                decoder_nbrs=decoder_graph_batch if len(batch) > 3 else None
            )
        else:
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord_batch,
                pndata=x_batch,
                encoder_nbrs=encoder_graph_batch if len(batch) > 3 else None,
                decoder_nbrs=decoder_graph_batch if len(batch) > 3 else None
            )
        
        return self.loss_fn(pred, y_batch)
    
    def validate(self, loader):
        """Validate the model on validation set."""
        if loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                if self.coord_mode == 'fx':
                    loss = self._validate_fixed_coords(batch)
                else:
                    loss = self._validate_variable_coords(batch)
                
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate_fixed_coords(self, batch):
        """Validation step for fixed coordinates."""
        x_batch, y_batch = batch
        
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        coord = self.coord.to(self.device)
        
        if getattr(self.model_config, 'use_conditional_norm', False):
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord,
                pndata=x_batch[..., :-1],
                condition=x_batch[..., 0, -2:-1]
            )
        else:
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord,
                pndata=x_batch
            )
        
        return self.loss_fn(pred, y_batch)
    
    def _validate_variable_coords(self, batch):
        """Validation step for variable coordinates."""
        if len(batch) == 3:
            x_batch, y_batch, coord_batch = batch
            encoder_graph_batch = decoder_graph_batch = None
        else:
            x_batch, y_batch, coord_batch, encoder_graph_batch, decoder_graph_batch = batch
            encoder_graph_batch = move_to_device(encoder_graph_batch, self.device)
            decoder_graph_batch = move_to_device(decoder_graph_batch, self.device)
        
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        coord_batch = coord_batch.to(self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        
        if getattr(self.model_config, 'use_conditional_norm', False):
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord_batch,
                pndata=x_batch[..., :-1],
                condition=x_batch[..., 0, -2:-1],
                encoder_nbrs=encoder_graph_batch,
                decoder_nbrs=decoder_graph_batch
            )
        else:
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord_batch,
                pndata=x_batch,
                encoder_nbrs=encoder_graph_batch,
                decoder_nbrs=decoder_graph_batch
            )
        
        return self.loss_fn(pred, y_batch)
    
    def _call_model_autoregressive_predict(self, x_batch, time_indices, coord_batch=None):
        """
        Call the model's autoregressive_predict method with appropriate parameters.
        
        Args:
            x_batch: Initial input batch at time t=0
            time_indices: Array of time indices for prediction
            coord_batch: Coordinate batch for variable coords mode
            
        Returns:
            Predicted outputs over time
        """
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        
        if self.coord_mode == 'fx':
            fixed_coord = self.coord.to(self.device)
            encoder_nbrs = None
            decoder_nbrs = None
        else:
            if coord_batch is None:
                raise ValueError("coord_batch is required for variable coordinates autoregressive prediction.")
            # Per-sample fixed coordinates (vx): compute neighbors on the fly when precompute_edges=False.
            fixed_coord = coord_batch.to(self.device)
            if fixed_coord.ndim == 4 and fixed_coord.shape[1] == 1:
                fixed_coord = fixed_coord[:, 0]
            if fixed_coord.ndim == 2:
                fixed_coord = fixed_coord.unsqueeze(0)
            if fixed_coord.ndim != 3:
                raise ValueError(f"coord_batch must be 2D or 3D after normalization, got {fixed_coord.shape}.")
            if fixed_coord.shape[1] != x_batch.shape[1]:
                raise ValueError(
                    f"coord_batch node count {fixed_coord.shape[1]} does not match input nodes {x_batch.shape[1]}."
                )
            encoder_nbrs = None
            decoder_nbrs = None
        
        return self.model.autoregressive_predict(
            x_batch=x_batch,
            time_indices=time_indices,
            t_values=self.t_values,
            stats=self.stats,
            stepper_mode=self.stepper_mode,
            latent_tokens_coord=latent_tokens_coord,
            fixed_coord=fixed_coord,
            encoder_nbrs=encoder_nbrs,
            decoder_nbrs=decoder_nbrs,
            use_conditional_norm=getattr(self.model_config, 'use_conditional_norm', False)
        )
    
    def test(self):
        """Test the model with different prediction modes."""
        print("Starting sequential model testing...")
        
        self.model.eval()
        self.model.to(self.device)
        
        if hasattr(self.dataset_config, 'predict_mode') and self.dataset_config.predict_mode == "all":
            modes = ["autoregressive", "direct", "star"]
        else:
            modes = [getattr(self.dataset_config, 'predict_mode', 'autoregressive')]
        
        errors_dict = {}
        example_data = None
        
        for mode in modes:
            print(f"Testing in {mode} mode...")
            all_relative_errors = []
            final_step_idx = min(
                len(self.t_values) - 1,
                self.test_loader.dataset.u_data.shape[1] - 1
            )

            if mode == "autoregressive":
                time_indices_list = [np.arange(0, final_step_idx + 1, dtype=np.int64)]
            elif mode == "direct":
                start_indices = np.arange(0, final_step_idx, 10, dtype=np.int64)
                if start_indices.size == 0:
                    start_indices = np.array([0], dtype=np.int64)
                time_indices_list = [
                    np.array([start_idx, final_step_idx], dtype=np.int64)
                    for start_idx in start_indices
                ]
            elif mode == "star":
                time_indices_list = [np.array([0, 4, 8, 12, 14], dtype=np.int64)]
            else:
                time_indices_list = [np.arange(0, final_step_idx + 1, dtype=np.int64)]

            test_sample_names = None
            if getattr(self.data_processor, "sample_names", None) is not None and getattr(self.data_processor, "split_indices", None) is not None:
                test_indices = self.data_processor.split_indices.get("test")
                if test_indices is not None:
                    test_sample_names = [
                        self._normalize_sample_id(self.data_processor.sample_names[idx])
                        for idx in test_indices
                    ]

            for time_indices in time_indices_list:
                test_dataset = TestDataset(
                    u_data=self.test_loader.dataset.u_data,
                    c_data=self.test_loader.dataset.c_data,
                    t_values=self.test_loader.dataset.t_values,
                    metadata=self.metadata,
                    time_indices=time_indices,
                    stats=self.stats,
                    x_data=self.test_loader.dataset.x_data,
                    is_variable_coords=(self.coord_mode == 'vx'),
                    zone_ids=self.test_loader.dataset.zone_id if hasattr(self.test_loader.dataset, "zone_id") else None,
                    sample_names=test_sample_names
                )

                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=self.dataset_config.batch_size,
                    shuffle=False,
                    num_workers=self.dataset_config.num_workers,
                    collate_fn=collate_sequential_batch
                )

                with torch.no_grad():
                    pbar = tqdm(total=len(test_loader), desc=f"Testing ({mode})", colour="blue")

                    for batch_idx, batch in enumerate(test_loader):
                        sample_ids = None
                        zone_ids = None
                        if self.coord_mode == 'fx':
                            x_batch, y_batch = batch
                            coord_batch = None
                        else:
                            if len(batch) == 3:
                                x_batch, y_batch, coord_batch = batch
                            elif len(batch) == 4:
                                x_batch, y_batch, coord_batch, sample_ids = batch
                            else:
                                x_batch, y_batch, coord_batch, sample_ids, zone_ids = batch

                        x_batch = x_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        pred = self._call_model_autoregressive_predict(
                            x_batch,
                            time_indices,
                            coord_batch if coord_batch is not None else None
                        )

                        sample_indices = self._resolve_sample_indices(
                            sample_ids, batch_idx, x_batch.shape[0]
                        )
                        pred_denorm = self._denormalize_temperature(pred, sample_indices)
                        y_denorm = self._denormalize_temperature(y_batch, sample_indices)

                        metric_type = getattr(self.dataset_config, 'metric', 'final_step')
                        normalize_metric = getattr(self.dataset_config, "metric_normalize", False)
                        if metric_type == "final_step":
                            y_err = y_batch[:, -1:, :, :] if normalize_metric else y_denorm[:, -1:, :, :]
                            pred_err = pred[:, -1:, :, :] if normalize_metric else pred_denorm[:, -1:, :, :]
                            relative_errors = compute_batch_errors(
                                y_err, pred_err, self.metadata, normalize=normalize_metric)
                        elif metric_type == "all_step":
                            y_err = y_batch if normalize_metric else y_denorm
                            pred_err = pred if normalize_metric else pred_denorm
                            relative_errors = compute_batch_errors(y_err, pred_err, self.metadata, normalize=normalize_metric)
                        else:
                            raise ValueError(f"Unknown metric: {metric_type}")

                        all_relative_errors.append(relative_errors)
                        pbar.update(1)

                        self._save_rollouts(
                            pred=pred_denorm,
                            y_batch=y_denorm,
                            x_batch=x_batch,
                            coord_batch=coord_batch,
                            time_indices=time_indices,
                            sample_ids=sample_ids,
                            sample_indices=sample_indices,
                            zone_ids=zone_ids if "zone_ids" in locals() else None,
                            mode=mode
                        )

                        if example_data is None:
                            coord_batch_for_plot = coord_batch if self.coord_mode == 'vx' else None
                            example_data = self._prepare_example_data(
                                x_batch, y_denorm, pred_denorm, time_indices, coord_batch_for_plot, sample_indices
                            )

                    pbar.close()
            
            all_relative_errors = torch.cat(all_relative_errors, dim=0)
            final_metric = compute_final_metric(all_relative_errors)
            errors_dict[mode] = final_metric
            print(f"{mode} mode error: {final_metric}")
        
        self._store_test_results(errors_dict, modes)
        
        if example_data is not None:
            self._plot_test_results(example_data)
            
            # TODO: Add animation for vx mode, currently only support fx mode
            if self.coord_mode == 'fx' and self.coord_dim == 2:
                self._create_animation(example_data)
        
        print("Sequential model testing complete.")
    
    def _prepare_example_data(self, x_batch, y_batch, pred, time_indices, coord_batch=None, sample_indices=None):
        """Prepare data for plotting."""
        u_dim = self.stats["u"]["mean"].shape[0]
        c_dim = self.stats["c"]["mean"].shape[0] if "c" in self.stats else 0
        if c_dim > 0:
            x_u_part = x_batch[..., :u_dim].cpu() * self.stats["u"]["std"] + self.stats["u"]["mean"]
            x_c_part = x_batch[..., u_dim:u_dim+c_dim].cpu() * self.stats["c"]["std"] + self.stats["c"]["mean"]
            x_u_denorm = self._denormalize_temperature(x_u_part, sample_indices).cpu()
            x_input = np.concatenate([x_u_denorm.numpy(), x_c_part.numpy()], axis=-1)
        else:
            x_u_part = x_batch[..., :u_dim].cpu() * self.stats["u"]["std"] + self.stats["u"]["mean"]
            x_u_denorm = self._denormalize_temperature(x_u_part, sample_indices).cpu()
            x_input = x_u_denorm.numpy()
        
        if self.coord_mode == 'fx':
            original_coords = self.data_processor.coord_scaler.inverse_transform(self.coord.cpu())
            coord_data = original_coords.numpy()
        else:
            if coord_batch is not None:
                original_coords = self.data_processor.coord_scaler.inverse_transform(coord_batch[-1].cpu())
                coord_data = original_coords.numpy()
            else:
                coord_data = None
        
        return {
            'input': x_input[-1],
            'coords': coord_data,
            'gt_sequence': y_batch[-1].cpu().numpy(),
            'pred_sequence': pred[-1].cpu().numpy(),
            'time_indices': time_indices,
            't_values': self.t_values
        }

    def _normalize_sample_id(self, sample_id) -> str:
        """Normalize sample id for filenames."""
        if isinstance(sample_id, bytes):
            sample_id = sample_id.decode("utf-8")
        sample_str = str(sample_id)
        sample_str = sample_str.replace("/", "_").replace(" ", "_")
        return sample_str

    def _resolve_sample_indices(self, sample_ids, batch_idx: int, batch_size: int) -> list[int]:
        """Resolve sample indices for the current batch."""
        if sample_ids is None:
            start = batch_idx * batch_size
            return list(range(start, start + batch_size))

        resolved = []
        name_to_idx = getattr(self.data_processor, "sample_name_to_idx", None)
        normalized_map = getattr(self, "_sample_name_to_idx_normalized", None)
        for sample_id in sample_ids:
            if isinstance(sample_id, bytes):
                sample_id = sample_id.decode("utf-8")
            if isinstance(sample_id, (int, np.integer)):
                resolved.append(int(sample_id))
                continue
            sample_str = str(sample_id)
            if name_to_idx and sample_str in name_to_idx:
                resolved.append(int(name_to_idx[sample_str]))
                continue
            if name_to_idx:
                if normalized_map is None:
                    normalized_map = {
                        self._normalize_sample_id(name): idx
                        for name, idx in name_to_idx.items()
                    }
                    self._sample_name_to_idx_normalized = normalized_map
                if sample_str in normalized_map:
                    resolved.append(int(normalized_map[sample_str]))
                    continue
            try:
                resolved.append(int(sample_str))
            except ValueError as exc:
                raise ValueError(f"Unknown sample id '{sample_str}'") from exc
        return resolved

    def _denormalize_temperature(self, data: torch.Tensor, sample_indices: list[int] | None) -> torch.Tensor:
        """Undo dataset temperature normalization modes."""
        temp_mode = getattr(self.data_processor, "temp_norm_mode", "none")
        if temp_mode == "none":
            return data

        if temp_mode == "global":
            mean = self.data_processor.temp_global_mean
            std = self.data_processor.temp_global_std
            if mean is None or std is None:
                raise ValueError("Missing temp_mean/temp_std for global normalization.")
            mean_t = torch.tensor(mean, device=data.device, dtype=data.dtype)
            std_t = torch.tensor(std, device=data.device, dtype=data.dtype)
            return data * std_t + mean_t

        if temp_mode == "per-sample":
            if sample_indices is None:
                raise ValueError("Sample indices required for per-sample denormalization.")
            means = self.data_processor.temp_sample_mean
            stds = self.data_processor.temp_sample_std
            if means is None or stds is None:
                raise ValueError("Missing temp_mean/temp_std for per-sample normalization.")
            if data.ndim == 3:
                view_shape = (-1, 1, 1)
            elif data.ndim == 4:
                view_shape = (-1, 1, 1, 1)
            else:
                raise ValueError(f"Unsupported data rank for per-sample denorm: {data.ndim}")
            mean_t = torch.tensor(means[sample_indices], device=data.device, dtype=data.dtype).view(*view_shape)
            std_t = torch.tensor(stds[sample_indices], device=data.device, dtype=data.dtype).view(*view_shape)
            return data * std_t + mean_t

        raise ValueError(f"Unsupported temp_norm_mode: {temp_mode}")

    def _save_rollouts(self, pred, y_batch, x_batch, coord_batch, time_indices, sample_ids, sample_indices, zone_ids, mode):
        """Save predicted and ground-truth sequences for each sample in the batch."""
        import os
        import h5py
        import numpy as np

        if not getattr(self.dataset_config, "save_rollouts", False):
            return
        if mode != "autoregressive":
            return

        output_dir = getattr(self.path_config, "rollout_dir", None)
        if not output_dir:
            output_dir = os.path.dirname(self.path_config.database_path)
        os.makedirs(output_dir, exist_ok=True)

        u_mean = self.stats["u"]["mean"].to(pred.device)
        u_std = self.stats["u"]["std"].to(pred.device)
        u_dim = u_mean.shape[0]

        u_start_norm = x_batch[..., :u_dim]
        u_start_denorm = u_start_norm * u_std + u_mean
        u_start_phys = self._denormalize_temperature(u_start_denorm, sample_indices)

        gt_full = torch.cat([u_start_phys[:, None, ...], y_batch], dim=1)
        pred_full = torch.cat([u_start_phys[:, None, ...], pred], dim=1)

        if self.coord_mode == 'fx':
            coords = self.coord.to(pred.device).unsqueeze(0).expand(x_batch.shape[0], -1, -1)
        else:
            coords = coord_batch.to(pred.device)
            if coords.ndim == 4 and coords.shape[1] == 1:
                coords = coords[:, 0]
        coords_np = coords.detach().cpu().numpy().astype(np.float32)

        dt_label = "0min"
        if len(time_indices) > 1:
            diffs = np.diff(self.t_values[time_indices])
            if np.allclose(diffs, diffs[0]):
                dt_value = float(diffs[0])
                dt_label = f"{int(round(dt_value))}min"
            else:
                dt_label = "var"
        t0_idx = int(time_indices[0])

        for i in range(x_batch.shape[0]):
            sample_id = sample_ids[i] if sample_ids is not None else i
            safe_id = self._normalize_sample_id(sample_id)
            filename = f"rollout_{safe_id}_t0={t0_idx}_dt={dt_label}.hdf5"
            path = os.path.join(output_dir, filename)

            temps = gt_full[i].detach().cpu().numpy().astype(np.float32)
            temps_pred = pred_full[i].detach().cpu().numpy().astype(np.float32)
            node_pos = coords_np[i]
            n_nodes = node_pos.shape[0]

            with h5py.File(path, "w") as f:
                f.create_dataset("node_pos", data=node_pos)
                f.create_dataset("temperatures", data=temps)
                f.create_dataset("temperatures_pred", data=temps_pred)
                if zone_ids is not None:
                    f.create_dataset("zone_id", data=zone_ids[i].astype(np.int32))

                zeros_f = np.zeros(n_nodes, dtype=np.float32)
                zeros_i = np.zeros(n_nodes, dtype=np.int32)
                f.create_dataset("Cp", data=zeros_f)
                f.create_dataset("M", data=zeros_f)
                f.create_dataset("Q", data=zeros_f)
                f.create_dataset("V", data=zeros_f)
                f.create_dataset("cp", data=zeros_f)
                f.create_dataset("cp_vol", data=zeros_f)
                f.create_dataset("k", data=zeros_f)
                f.create_dataset("q", data=zeros_f)
                f.create_dataset("rho", data=zeros_f)
                f.create_dataset("node_group", data=zeros_i)
                f.create_dataset("node_types", data=zeros_i)
                f.create_dataset("edge_src", data=np.zeros(0, dtype=np.int32))
                f.create_dataset("edge_dst", data=np.zeros(0, dtype=np.int32))
                f.create_dataset("edge_area", data=np.zeros(0, dtype=np.float32))
    
    def _store_test_results(self, errors_dict, modes):
        """Store test results in config datarow."""
        if len(modes) > 1:  
            self.config.datarow["relative error (direct)"] = errors_dict.get("direct", 0.0)
            self.config.datarow["relative error (auto2)"] = errors_dict.get("autoregressive", 0.0)
            self.config.datarow["relative error (auto4)"] = errors_dict.get("star", 0.0)
        else:  
            mode = modes[0]
            self.config.datarow[f"relative error ({mode})"] = errors_dict[mode]
    
    def _plot_test_results(self, example_data):
        """Create and save test result plots."""
        try:
            if getattr(self.dataset_config, "plot_c_in_tests", False) and self.metadata.names['c'] and 'c' in self.stats:
                # Plot with condition data
                u_dim = len(self.metadata.names['u'])
                input_plot = example_data['input'][..., :u_dim].squeeze(1)
                c_names = self.metadata.names['c']
                c_signed = self.metadata.signed['c']
                c_index = -1
                if isinstance(c_names, list) and c_names:
                    c_name = c_names[c_index]
                    c_sign = c_signed[c_index] if isinstance(c_signed, list) else c_signed
                else:
                    c_name = "c"
                    c_sign = False
                c_channel = example_data['input'][..., -1]
                if c_channel.ndim == 1:
                    c_channel = c_channel[:, None]
                c_channel = c_channel.squeeze(1)
                gt_with_c = np.concatenate(
                    [example_data['gt_sequence'][-1], c_channel], axis=-1
                ).squeeze(1)
                pred_with_c = np.concatenate(
                    [example_data['pred_sequence'][-1], c_channel], axis=-1
                ).squeeze(1)
                
                fig = plot_estimates(
                    u_inp=input_plot,
                    u_gtr=gt_with_c,
                    u_prd=pred_with_c,
                    x_inp=example_data['coords'],
                    x_out=example_data['coords'],
                    names=self.metadata.names['u'] + [c_name],
                    symmetric=self.metadata.signed['u'] + [c_sign],
                    domain=self.metadata.domain_x
                )
            else:
                # Plot without condition data
                fig = plot_estimates(
                    u_inp=example_data['input'],
                    u_gtr=example_data['gt_sequence'][-1],
                    u_prd=example_data['pred_sequence'][-1],
                    x_inp=example_data['coords'],
                    x_out=example_data['coords'],
                    names=self.metadata.names['u'],
                    symmetric=self.metadata.signed['u'],
                    domain=self.metadata.domain_x
                )
            
            fig.savefig(self.path_config.result_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            print(f"Plot saved to {self.path_config.result_path}")
            
            import matplotlib.pyplot as plt
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not create plot: {e}")
    
    def _create_animation(self, example_data):
        """Create and save animation for sequential data."""
        try:
            animation_path = self.path_config.result_path.replace('.png', '.gif')
            gt_sequence = example_data['gt_sequence']      # [n_timesteps, n_points, n_channels]
            pred_sequence = example_data['pred_sequence']  # [n_timesteps, n_points, n_channels]
            coords = example_data['coords']                # [n_points, coord_dim]
            time_indices = example_data['time_indices']
            t_values_full = example_data['t_values']
            
            if len(time_indices) > 1:
                time_values = [t_values_full[idx] for idx in time_indices[1:]]  # Skip initial time
            else:
                time_values = [f"Step {i}" for i in range(len(gt_sequence))]
            
            create_sequential_animation(
                gt_sequence=gt_sequence,
                pred_sequence=pred_sequence,
                coords=coords,
                save_path=animation_path,
                input_data=example_data['input'],  
                time_values=time_values,
                interval=800,  
                symmetric=self.metadata.signed['u'] if self.metadata.signed.get('u') else [True],
                domain=self.metadata.domain_x if hasattr(self.metadata, 'domain_x') else None,
                names=self.metadata.names['u'] if self.metadata.names.get('u') else None,
                colorbar_type="light",
                show_error=True
            )
            
            print(f"Animation saved to {animation_path}")
            
        except Exception as e:
            print(f"Warning: Could not create animation: {e}")
            import traceback
            traceback.print_exc()
