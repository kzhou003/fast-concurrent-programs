"""
PROBLEM: Implement KV Cache Manager for LLM Inference

Build a cache management system for Key-Value caches during LLM inference.
KV caches are critical for efficient token generation, but memory is limited.
This system manages allocation, eviction, and sharing of cached values.

REQUIREMENTS:
- Track KV cache for each sequence (growing with each token)
- Implement memory tracking and allocation
- Support multiple eviction policies (LRU, LFU, random)
- Reuse cache blocks between sequences (important for efficiency)
- Handle variable sequence lengths
- Track memory usage and fragmentation

PERFORMANCE NOTES:
- Should minimize memory fragmentation
- Cache hits should be common for good performance
- Eviction decisions should be fast (not block inference)
- Should support 1000+ concurrent sequences

TEST CASE EXPECTATIONS:
- Cache should grow as sequences are extended
- Memory should be properly deallocated on eviction
- Eviction policies should work correctly
- Cache reuse should reduce memory usage
- Memory limits should be enforced
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import OrderedDict
import time


@dataclass
class CacheBlock:
    """Represents a contiguous block of KV cache memory."""

    block_id: int
    size: int  # Number of tokens this block can hold
    allocated: bool = False
    sequence_id: Optional[int] = None
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0

    def mark_access(self):
        """Mark this block as accessed."""
        self.last_access_time = time.time()
        self.access_count += 1


class KVCacheManager:
    """Manages KV cache allocation and eviction."""

    def __init__(
        self,
        total_cache_size: int = 1000000,  # Total number of tokens worth of cache
        block_size: int = 128,  # Size of each block
        eviction_policy: str = "lru",  # lru, lfu, random
        num_heads: int = 32,
        hidden_dim: int = 128,
    ):
        """
        Initialize KV cache manager.

        Args:
            total_cache_size: Total cache size in tokens
            block_size: Size of each cache block in tokens
            eviction_policy: Eviction policy (lru, lfu, random)
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension per head
        """
        self.total_cache_size = total_cache_size
        self.block_size = block_size
        self.eviction_policy = eviction_policy
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Bytes per token: 2 * num_heads * hidden_dim * 2 (for K and V, float32)
        self.bytes_per_token = 2 * num_heads * hidden_dim * 4

        # Total memory in bytes
        self.total_memory = total_cache_size * self.bytes_per_token

        # Initialize free blocks
        self.free_blocks: OrderedDict[int, CacheBlock] = OrderedDict()
        self.allocated_blocks: Dict[int, CacheBlock] = {}
        self.sequence_to_blocks: Dict[int, List[int]] = {}

        self.next_block_id = 0
        self.used_memory = 0
        self.next_sequence_id = 0

    def _create_initial_blocks(self):
        """Create initial free blocks."""
        # TODO: Initialize free blocks by dividing total cache
        num_blocks = self.total_cache_size // self.block_size
        for i in range(num_blocks):
            block = CacheBlock(block_id=self.next_block_id, size=self.block_size)
            self.next_block_id += 1
            self.free_blocks[block.block_id] = block

    def allocate_blocks(self, sequence_id: int, num_tokens: int) -> List[int]:
        """
        Allocate cache blocks for a sequence.

        Args:
            sequence_id: ID of the sequence
            num_tokens: Number of tokens to allocate cache for

        Returns:
            List of block IDs allocated
        """
        # TODO: Implement block allocation
        # 1. Calculate number of blocks needed (round up num_tokens / block_size)
        # 2. Find free blocks, allocate as needed
        # 3. If not enough free blocks, evict blocks per policy
        # 4. Track allocation in sequence_to_blocks
        allocated_blocks = []

        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        # Get free blocks
        while len(allocated_blocks) < blocks_needed and len(self.free_blocks) > 0:
            block_id, block = self.free_blocks.popitem(last=False)
            block.allocated = True
            block.sequence_id = sequence_id
            self.allocated_blocks[block_id] = block
            allocated_blocks.append(block_id)

        # If still need blocks, evict
        while len(allocated_blocks) < blocks_needed:
            evicted_block_id = self._evict_block()
            if evicted_block_id is None:
                raise RuntimeError("Cannot allocate cache: no blocks available")

            block = self.allocated_blocks[evicted_block_id]
            block.sequence_id = sequence_id
            allocated_blocks.append(evicted_block_id)

        self.sequence_to_blocks[sequence_id] = allocated_blocks
        self.used_memory += len(allocated_blocks) * self.block_size * self.bytes_per_token

        return allocated_blocks

    def _evict_block(self) -> Optional[int]:
        """
        Evict a block according to eviction policy.

        Returns:
            Block ID of evicted block, or None if no blocks to evict
        """
        # TODO: Implement eviction based on policy
        if not self.allocated_blocks:
            return None

        if self.eviction_policy == "lru":
            # Least Recently Used
            block_id = min(
                self.allocated_blocks.keys(),
                key=lambda bid: self.allocated_blocks[bid].last_access_time,
            )
        elif self.eviction_policy == "lfu":
            # Least Frequently Used
            block_id = min(
                self.allocated_blocks.keys(),
                key=lambda bid: self.allocated_blocks[bid].access_count,
            )
        elif self.eviction_policy == "random":
            block_id = next(iter(self.allocated_blocks))
        else:
            block_id = next(iter(self.allocated_blocks))

        # Remove from allocated and add to free
        block = self.allocated_blocks.pop(block_id)
        if block.sequence_id in self.sequence_to_blocks:
            self.sequence_to_blocks[block.sequence_id].remove(block_id)

        block.allocated = False
        block.sequence_id = None
        self.free_blocks[block_id] = block

        return block_id

    def extend_cache(self, sequence_id: int, new_length: int):
        """
        Extend cache for a sequence.

        Args:
            sequence_id: ID of the sequence
            new_length: New total length of sequence
        """
        # TODO: Implement cache extension
        # If sequence already has blocks, check if need more
        # If need more blocks, allocate
        if sequence_id not in self.sequence_to_blocks:
            self.allocate_blocks(sequence_id, new_length)
        else:
            current_capacity = len(self.sequence_to_blocks[sequence_id]) * self.block_size
            if new_length > current_capacity:
                # Need more blocks
                additional_tokens = new_length - current_capacity
                new_blocks = self.allocate_blocks(sequence_id, additional_tokens)
                self.sequence_to_blocks[sequence_id].extend(new_blocks)

    def get_cache_size(self, sequence_id: int) -> int:
        """Get total cache size allocated for sequence."""
        if sequence_id not in self.sequence_to_blocks:
            return 0
        return len(self.sequence_to_blocks[sequence_id]) * self.block_size

    def release_sequence(self, sequence_id: int):
        """
        Release all cache blocks for a sequence.

        Args:
            sequence_id: ID of the sequence
        """
        # TODO: Implement release
        # Mark all blocks for this sequence as free
        if sequence_id in self.sequence_to_blocks:
            for block_id in self.sequence_to_blocks[sequence_id]:
                if block_id in self.allocated_blocks:
                    block = self.allocated_blocks.pop(block_id)
                    block.allocated = False
                    block.sequence_id = None
                    self.free_blocks[block_id] = block

            self.used_memory -= len(self.sequence_to_blocks[sequence_id]) * self.block_size * self.bytes_per_token
            del self.sequence_to_blocks[sequence_id]

    def memory_usage(self) -> float:
        """Return memory usage as percentage of total."""
        return (self.used_memory / self.total_memory) * 100

    def get_free_blocks(self) -> int:
        """Return number of free blocks."""
        return len(self.free_blocks)


def test_basic_allocation():
    """Test basic cache allocation."""
    manager = KVCacheManager(total_cache_size=1000, block_size=128)
    manager._create_initial_blocks()

    # Allocate cache for sequence
    seq_id = 0
    blocks = manager.allocate_blocks(seq_id, 256)

    assert len(blocks) == 2  # 256 / 128 = 2 blocks
    assert manager.get_cache_size(seq_id) == 256
    assert manager.memory_usage() > 0

    print(f"✓ Basic allocation test passed (allocated {len(blocks)} blocks)")


def test_memory_limit():
    """Test that total memory is not exceeded."""
    manager = KVCacheManager(total_cache_size=1000, block_size=128)
    manager._create_initial_blocks()

    # Allocate all available memory
    for i in range(100):
        try:
            blocks = manager.allocate_blocks(i, 10)
        except RuntimeError:
            break

    # Should not exceed 100% (within block granularity)
    usage = manager.memory_usage()
    assert usage <= 100 + 10, f"Memory usage {usage}% exceeds 100%"

    print(f"✓ Memory limit test passed ({usage:.1f}% used)")


def test_lru_eviction():
    """Test LRU eviction policy."""
    manager = KVCacheManager(
        total_cache_size=500, block_size=128, eviction_policy="lru"
    )
    manager._create_initial_blocks()

    # Allocate full cache
    manager.allocate_blocks(0, 500)

    # Now try to allocate more (will evict)
    initial_free = manager.get_free_blocks()

    # Access block to update LRU
    if 0 in manager.sequence_to_blocks:
        block_id = manager.sequence_to_blocks[0][0]
        if block_id in manager.allocated_blocks:
            manager.allocated_blocks[block_id].mark_access()

    # Allocate for new sequence
    manager.allocate_blocks(1, 128)

    # Some blocks should have been evicted
    assert manager.get_free_blocks() < initial_free

    print(f"✓ LRU eviction test passed")


def test_lfu_eviction():
    """Test LFU eviction policy."""
    manager = KVCacheManager(
        total_cache_size=500, block_size=128, eviction_policy="lfu"
    )
    manager._create_initial_blocks()

    # Allocate and access some blocks
    manager.allocate_blocks(0, 256)

    # Allocate and fill
    manager.allocate_blocks(1, 256)

    # Access sequence 0 multiple times (should not be evicted)
    for block_id in manager.sequence_to_blocks[0]:
        manager.allocated_blocks[block_id].mark_access()

    # Try to allocate more
    try:
        manager.allocate_blocks(2, 128)
    except RuntimeError:
        pass

    # Sequence 0 should still be there (most accessed)
    assert len(manager.sequence_to_blocks[0]) > 0

    print(f"✓ LFU eviction test passed")


def test_cache_extension():
    """Test extending cache for existing sequence."""
    manager = KVCacheManager(total_cache_size=1000, block_size=128)
    manager._create_initial_blocks()

    # Allocate initial cache
    seq_id = 0
    initial_blocks = manager.allocate_blocks(seq_id, 256)

    # Extend cache
    manager.extend_cache(seq_id, 512)

    # Should have more blocks now
    extended_size = manager.get_cache_size(seq_id)
    assert extended_size >= 512

    print(f"✓ Cache extension test passed (extended to {extended_size})")


def test_release_sequence():
    """Test releasing cache for sequence."""
    manager = KVCacheManager(total_cache_size=1000, block_size=128)
    manager._create_initial_blocks()

    # Allocate for multiple sequences
    manager.allocate_blocks(0, 256)
    manager.allocate_blocks(1, 256)

    initial_usage = manager.memory_usage()

    # Release sequence 0
    manager.release_sequence(0)

    final_usage = manager.memory_usage()

    assert final_usage < initial_usage
    assert 0 not in manager.sequence_to_blocks

    print(f"✓ Release sequence test passed (freed {initial_usage - final_usage:.1f}%)")


def test_multiple_sequences():
    """Test managing cache for multiple concurrent sequences."""
    manager = KVCacheManager(total_cache_size=2000, block_size=128)
    manager._create_initial_blocks()

    # Allocate for multiple sequences
    sequences = {}
    for i in range(5):
        length = 100 + i * 50
        blocks = manager.allocate_blocks(i, length)
        sequences[i] = manager.get_cache_size(i)

    # All sequences should have cache
    assert all(size > 0 for size in sequences.values())

    # Total usage should be reasonable
    usage = manager.memory_usage()
    assert usage < 100

    print(f"✓ Multiple sequences test passed (5 sequences, {usage:.1f}% usage)")


def test_fragmentation():
    """Test memory fragmentation with allocations and deallocations."""
    manager = KVCacheManager(total_cache_size=1000, block_size=128)
    manager._create_initial_blocks()

    # Allocate and release alternately
    for i in range(10):
        manager.allocate_blocks(i, 100)

    for i in range(5):
        manager.release_sequence(i)

    # Should be able to allocate again
    new_blocks = manager.allocate_blocks(100, 200)
    assert len(new_blocks) > 0

    print(f"✓ Fragmentation test passed")


def test_different_sequence_lengths():
    """Test with highly variable sequence lengths."""
    manager = KVCacheManager(total_cache_size=5000, block_size=128)
    manager._create_initial_blocks()

    lengths = [10, 100, 500, 50, 1000, 25]

    for i, length in enumerate(lengths):
        blocks = manager.allocate_blocks(i, length)
        expected_size = ((length + 127) // 128) * 128
        assert manager.get_cache_size(i) == expected_size

    print(f"✓ Variable sequence lengths test passed")


if __name__ == "__main__":
    print("Running KV Cache Manager tests...\n")

    test_basic_allocation()
    test_memory_limit()
    test_lru_eviction()
    test_lfu_eviction()
    test_cache_extension()
    test_release_sequence()
    test_multiple_sequences()
    test_fragmentation()
    test_different_sequence_lengths()

    print("\n✓ All tests passed!")
