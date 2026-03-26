# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class KVLayerGroupKind(str, Enum):
    ATTENTION = "attention"
    GDN = "gdn"

@dataclass(frozen=True)
class KVLayerTensorSpec:
    name: str
    shape: torch.Size
    dtype: torch.dtype

@dataclass
class KVLayerGroupInfo:
    """Information about a group of layers with the same KV cache structure.

    Layers within the same group have identical shape and dtype for their KV cache.
    Different groups may have different shapes (especially head_size) and/or dtypes.
    """

    """ List of layer names belonging to this group """
    layer_names: list[str]
    """ List of layer indices (0-based) belonging to this group """
    layer_indices: list[int]
    """ Shape of the KV cache tensor for layers in this group """
    """ For MHA: typically [2, num_blocks, block_size, num_heads, head_size] """
    """ For MLA: typically [num_blocks, block_size, head_size] """
    shape: Optional[torch.Size] = None
    """ Data type of the KV cache tensor for layers in this group """
    dtype: Optional[torch.dtype] = None
    """ Logical group kind. """
    group_kind: KVLayerGroupKind = KVLayerGroupKind.ATTENTION
    """ Tensor layout specs for this group. """
    tensor_specs: list[KVLayerTensorSpec] = field(default_factory=list)

    # Internal sets for fast membership checking
    _layer_indices_set: set[int] = field(init=False, repr=False)
    _layer_names_set: set[str] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize sets for fast membership checking."""
        if not self.tensor_specs:
            if self.shape is None or self.dtype is None:
                raise ValueError(
                    "Either tensor_specs or both shape/dtype must be provided."
                )
            self.tensor_specs = [
                KVLayerTensorSpec(
                    name="primary",
                    shape=torch.Size(self.shape),
                    dtype=self.dtype,
                )
            ]
        else:
            self.tensor_specs = list(self.tensor_specs)
            if self.shape is None:
                self.shape = self.tensor_specs[0].shape
            if self.dtype is None:
                self.dtype = self.tensor_specs[0].dtype

        assert self.shape is not None
        assert self.dtype is not None
        self._layer_indices_set = set(self.layer_indices)
        self._layer_names_set = set(self.layer_names)

    def __repr__(self) -> str:
        if not self.layer_indices:
            indices_repr = "[]"
        else:
            indices_repr = f"{self.layer_indices[0]}-{self.layer_indices[-1]}"
        return (
            f"KVLayerGroupInfo(layers={len(self.layer_names)}, "
            f"indices={indices_repr}, "
            f"kind={self.group_kind}, "
            f"shape={self.shape}, dtype={self.dtype}, "
            f"num_tensors={self.num_tensors})"
        )

    @property
    def num_layers(self) -> int:
        """Return the number of layers in this group."""
        return len(self.layer_names)

    @property
    def primary_tensor_spec(self) -> KVLayerTensorSpec:
        return self.tensor_specs[0]

    @property
    def num_tensors(self) -> int:
        return len(self.tensor_specs)

    @property
    def tensor_shapes(self) -> list[torch.Size]:
        return [tensor_spec.shape for tensor_spec in self.tensor_specs]

    @property
    def tensor_dtypes(self) -> list[torch.dtype]:
        return [tensor_spec.dtype for tensor_spec in self.tensor_specs]

    @property
    def hidden_dim_size(self) -> int:
        """Return the size of the hidden dimension in this group."""
        # hidden_dim_size = num_heads * head_size
        if self.shape is None:
            raise ValueError("Group shape is not initialized")
        if len(self.shape) == 5:
            # MHA
            return self.shape[3] * self.shape[4]
        elif len(self.shape) == 3:
            # MLA
            return self.shape[2]
        else:
            raise ValueError(f"Invalid shape: {self.shape}")

    def contains_layer(self, layer_idx: int) -> bool:
        """Check if a layer index belongs to this group."""
        return layer_idx in self._layer_indices_set

    def contains_layer_name(self, layer_name: str) -> bool:
        """Check if a layer name belongs to this group."""
        return layer_name in self._layer_names_set


@dataclass
class KVLayerGroupsManager:
    """Manager for KV layer groups with the same structure.

    This class encapsulates the functionality for managing groups of layers
    that have identical KV cache structure (shape and dtype).
    """

    kv_layer_groups: list[KVLayerGroupInfo] = field(default_factory=list)

    @property
    def num_groups(self) -> int:
        """Return the number of KV layer groups."""
        return len(self.kv_layer_groups)

    def get_group_by_layer_idx(self, layer_idx: int) -> Optional[KVLayerGroupInfo]:
        """Get the KVLayerGroupInfo for a given layer index.

        Args:
            layer_idx: The 0-based index of the layer.

        Returns:
            The KVLayerGroupInfo containing this layer, or None if not found.
        """
        for group in self.kv_layer_groups:
            if group.contains_layer(layer_idx):
                return group
        return None

    def get_group_by_layer_name(self, layer_name: str) -> Optional[KVLayerGroupInfo]:
        """Get the KVLayerGroupInfo for a given layer name.

        Args:
            layer_name: The name of the layer.

        Returns:
            The KVLayerGroupInfo containing this layer, or None if not found.
        """
        for group in self.kv_layer_groups:
            if group.contains_layer_name(layer_name):
                return group
        return None

    def get_layer_shape(self, layer_idx: int) -> Optional[torch.Size]:
        """Get the shape of the KV cache for a given layer index.

        Args:
            layer_idx: The 0-based index of the layer.

        Returns:
            The shape, or None if layer not found.
        """
        group = self.get_group_by_layer_idx(layer_idx)
        return group.shape if group else None

    def get_layer_dtype(self, layer_idx: int) -> Optional[torch.dtype]:
        """Get the dtype of the KV cache for a given layer index.

        Args:
            layer_idx: The 0-based index of the layer.

        Returns:
            The dtype, or None if layer not found.
        """
        group = self.get_group_by_layer_idx(layer_idx)
        return group.dtype if group else None

    def get_group_kind(self, group_idx: int) -> KVLayerGroupKind:
        return self.kv_layer_groups[group_idx].group_kind

    def group_has_multiple_tensors(self, group_idx: int) -> bool:
        return self.kv_layer_groups[group_idx].num_tensors > 1

    def _infer_group_kind_and_tensor_specs(
        self,
        kv_cache: torch.Tensor | Sequence[torch.Tensor],
    ) -> tuple[KVLayerGroupKind, list[KVLayerTensorSpec]]:
        if isinstance(kv_cache, torch.Tensor):
            return (
                KVLayerGroupKind.ATTENTION,
                [
                    KVLayerTensorSpec(
                        name="kv",
                        shape=kv_cache.shape,
                        dtype=kv_cache.dtype,
                    )
                ],
            )

        if not isinstance(kv_cache, Sequence) or len(kv_cache) == 0:
            raise RuntimeError(f"Unknown KVCache type: {type(kv_cache)}")

        tensors = list(kv_cache)
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise RuntimeError(f"Unknown KVCache element type: {type(kv_cache)}")

        is_attention_pair = (
            len(tensors) == 2
            and tensors[0].shape == tensors[1].shape
            and tensors[0].dtype == tensors[1].dtype
            and tensors[0].ndim >= 4
        )
        group_kind = (
            KVLayerGroupKind.ATTENTION if is_attention_pair else KVLayerGroupKind.GDN
        )
        names = ["k", "v"] if is_attention_pair else (
            ["conv_state", "ssm_state"]
            if len(tensors) == 2
            else [f"tensor_{idx}" for idx in range(len(tensors))]
        )
        tensor_specs = [
            KVLayerTensorSpec(name=name, shape=tensor.shape, dtype=tensor.dtype)
            for name, tensor in zip(names, tensors, strict=True)
        ]
        return group_kind, tensor_specs

    def build_kv_layer_groups(
        self,
        kv_caches: dict[str, torch.Tensor | Sequence[torch.Tensor]],
    ) -> None:
        """Build KV layer groups structure by analyzing each layer's shape and dtype.

        Layers with the same shape and dtype are grouped together. This is useful
        because different layers may have different structures (especially the
        last dimension head_size may differ between groups), and different groups
        may have different dtypes.

        If layer groups are already built (non-empty list), this method does nothing.

        Args:
            kv_caches: Dictionary mapping layer names to KV cache tensors.
        """
        # Skip if already built (non-empty list)
        if len(self.kv_layer_groups) > 0:
            return

        if len(kv_caches) == 0:
            logger.debug("No KV caches available, skipping KV layer groups building")
            return

        # Group layers by logical KV layout in a single loop.
        groups_dict: dict[
            tuple[
                KVLayerGroupKind,
                tuple[tuple[str, torch.Size, torch.dtype], ...],
            ],
            list[tuple[str, int]],
        ] = defaultdict(list)
        group_specs: dict[
            tuple[
                KVLayerGroupKind,
                tuple[tuple[str, torch.Size, torch.dtype], ...],
            ],
            list[KVLayerTensorSpec],
        ] = {}

        for idx, (layer_name, kv_cache) in enumerate(kv_caches.items()):
            group_kind, tensor_specs = self._infer_group_kind_and_tensor_specs(kv_cache)
            key = (
                group_kind,
                tuple(
                    (tensor_spec.name, tensor_spec.shape, tensor_spec.dtype)
                    for tensor_spec in tensor_specs
                ),
            )
            group_specs[key] = tensor_specs
            groups_dict[key].append((layer_name, idx))

        # Build KVLayerGroupInfo list
        # Sort groups by the first layer index to maintain order
        def _get_first_layer_index(shape_dtype_key):
            """Get the index of the first layer in a layer group."""
            layer_group = groups_dict[
                shape_dtype_key
            ]  # list of (layer_name, layer_index) tuples
            first_layer_info = layer_group[0]  # first (layer_name, layer_index) tuple
            layer_index = first_layer_info[1]  # extract the layer index
            return layer_index

        sorted_keys = sorted(groups_dict.keys(), key=_get_first_layer_index)

        kv_layer_groups: list[KVLayerGroupInfo] = []
        for key in sorted_keys:
            layers = groups_dict[key]
            layer_names, layer_indices = zip(*layers, strict=False)
            tensor_specs = group_specs[key]

            group_info = KVLayerGroupInfo(
                layer_names=list(layer_names),
                layer_indices=list(layer_indices),
                shape=tensor_specs[0].shape,
                dtype=tensor_specs[0].dtype,
                group_kind=key[0],
                tensor_specs=tensor_specs,
            )
            kv_layer_groups.append(group_info)

        # Store the built groups
        self.kv_layer_groups = kv_layer_groups

        # Print the group structure
        logger.info("KV layer groups: %s", kv_layer_groups)
