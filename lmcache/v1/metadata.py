# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager

logger = init_logger(__name__)


@dataclass
class LMCacheMetadata:
    """
    LMCacheMetadata should be extracted from the northbound
    serving engine configuration and wrap the extraction of
    attributes (e.g. model name, tp rank, etc.)
    """

    """name of the LLM model"""
    model_name: str
    """ global world size when running under a distributed setting 
    (total number of workers)"""
    world_size: int
    """ host world size (workers on active localhost)
    This information can be useful for multi-node
    deployment. Will be the same as world_size 
    in single-node deployments.
    """
    local_world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int
    """ host worker id (a gpu bound worker id on active localhost)
    This information can be useful for multi-node deployment. 
    Will be the same as worker_id in single-node deployments.
    """
    local_worker_id: int
    """ the data type of kv tensors """
    # (Deprecated) Will be replaced by kv_layer_groups_manager in the future
    kv_dtype: torch.dtype
    """ the shape of kv tensors """
    # (Deprecated) Will be replaced by kv_layer_groups_manager in the future
    """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
    kv_shape: tuple[int, int, int, int, int]
    """ whether use MLA"""
    use_mla: bool = False
    """ the role of the current instance (e.g., 'scheduler', 'worker') """
    role: Optional[str] = None
    """ the first rank of the distributed setting """
    # TODO(baoloongmao): first_rank should be configurable
    first_rank = 0
    served_model_name: Optional[str] = None
    """chunk size"""
    chunk_size: int = 256
    """ Manager for groups of layers with identical KV cache structure """
    kv_layer_groups_manager: KVLayerGroupsManager = field(
        default_factory=KVLayerGroupsManager
    )
    """ engine_id for RPC path (used by lookup client/server) """
    engine_id: Optional[str] = None
    """ extra config from kv_connector (e.g., lmcache_rpc_port) """
    kv_connector_extra_config: Optional[dict] = None

    def is_first_rank(self) -> bool:
        """Check if the current worker is the first rank"""
        return self.worker_id == self.first_rank

    def _build_group_memory_shape(
        self,
        group: KVLayerGroupInfo,
        num_tokens: int,
    ) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size(
            [
                kv_size,
                group.num_layers,
                num_tokens,
                group.hidden_dim_size,
            ]
        )

    def get_group_kinds(self) -> list[str]:
        if self.kv_layer_groups_manager.kv_layer_groups:
            return [
                group.group_kind.value
                for group in self.kv_layer_groups_manager.kv_layer_groups
            ]
        return ["attention"]

    def get_group_tensor_dtypes(self) -> list[list[torch.dtype]]:
        if self.kv_layer_groups_manager.kv_layer_groups:
            return [
                group.tensor_dtypes
                for group in self.kv_layer_groups_manager.kv_layer_groups
            ]
        return [[self.kv_dtype]]

    def get_group_tensor_shapes(
        self,
        num_tokens: Optional[int] = None,
    ) -> list[list[torch.Size]]:
        del num_tokens
        if self.kv_layer_groups_manager.kv_layer_groups:
            return [
                group.tensor_shapes
                for group in self.kv_layer_groups_manager.kv_layer_groups
            ]
        return [[torch.Size(self.kv_shape)]]

    def group_has_multiple_tensors(self, group_idx: int) -> bool:
        if self.kv_layer_groups_manager.kv_layer_groups:
            return self.kv_layer_groups_manager.group_has_multiple_tensors(group_idx)
        return False

    def has_gdn_groups(self) -> bool:
        return any(group_kind == "gdn" for group_kind in self.get_group_kinds())

    def _build_gdn_transfer_shape(
        self,
        group: KVLayerGroupInfo,
        tensor_idx: int,
    ) -> torch.Size:
        tensor_spec = group.tensor_specs[tensor_idx]
        if len(tensor_spec.shape) < 2:
            raise ValueError(
                "GDN runtime tensor shape must have at least 2 dims, got "
                f"{tensor_spec.shape}"
            )
        return torch.Size([group.num_layers, *tensor_spec.shape[1:]])

    def get_group_transfer_shapes(
        self,
        num_tokens: Optional[int] = None,
    ) -> list[list[torch.Size]]:
        if num_tokens is None:
            num_tokens = self.chunk_size
        if self.kv_layer_groups_manager.kv_layer_groups:
            group_shapes: list[list[torch.Size]] = []
            for group in self.kv_layer_groups_manager.kv_layer_groups:
                if group.group_kind.value == "gdn":
                    group_shapes.append(
                        [
                            self._build_gdn_transfer_shape(group, tensor_idx)
                            for tensor_idx in range(group.num_tensors)
                        ]
                    )
                else:
                    group_shapes.append(
                        [self._build_group_memory_shape(group, num_tokens)]
                    )
            return group_shapes
        return [self.get_shapes(num_tokens)]

    def get_group_transfer_dtypes(self) -> list[list[torch.dtype]]:
        if self.kv_layer_groups_manager.kv_layer_groups:
            group_dtypes: list[list[torch.dtype]] = []
            for group in self.kv_layer_groups_manager.kv_layer_groups:
                if group.group_kind.value == "gdn":
                    group_dtypes.append(group.tensor_dtypes)
                else:
                    assert group.dtype is not None
                    group_dtypes.append([group.dtype])
            return group_dtypes
        return [self.get_dtypes()]

    def get_transfer_shapes(
        self,
        num_tokens: Optional[int] = None,
    ) -> list[torch.Size]:
        return [
            shape
            for group_shapes in self.get_group_transfer_shapes(num_tokens)
            for shape in group_shapes
        ]

    def get_transfer_dtypes(self) -> list[torch.dtype]:
        return [
            dtype
            for group_dtypes in self.get_group_transfer_dtypes()
            for dtype in group_dtypes
        ]

    def get_shapes(self, num_tokens: Optional[int] = None) -> list[torch.Size]:
        """Get the shapes of the KV cache in LMCache"""
        if num_tokens is None:
            num_tokens = self.chunk_size
        if self.kv_layer_groups_manager.kv_layer_groups:
            return [
                self._build_group_memory_shape(group, num_tokens)
                for group in self.kv_layer_groups_manager.kv_layer_groups
            ]
        return [
            torch.Size(
                [
                    self.kv_shape[1],
                    self.kv_shape[0],
                    num_tokens,
                    self.kv_shape[3] * self.kv_shape[4],
                ]
            )
        ]

    # TODO(chunxiaozheng): some uts do not `build_kv_layer_groups`
    def get_dtypes(self) -> list[torch.dtype]:
        if self.kv_layer_groups_manager.kv_layer_groups:
            return [
                group.dtype for group in self.kv_layer_groups_manager.kv_layer_groups
            ]
        return [self.kv_dtype]

    def get_num_groups(self) -> int:
        if self.kv_layer_groups_manager.kv_layer_groups:
            return self.kv_layer_groups_manager.num_groups
        return 1
