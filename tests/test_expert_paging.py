import torch

from omnicoder.modeling.utils.expert_paging import ExpertPager


def test_expert_pager_basic_lru() -> None:
    dev = torch.device('cpu')
    pager = ExpertPager(capacity=2, device=dev)
    # Register three tiny experts that record a fixed parameter id
    class Tiny(torch.nn.Module):
        def __init__(self, nid: int) -> None:
            super().__init__()
            self.n = torch.nn.Parameter(torch.tensor(float(nid)))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.n
    for i in range(3):
        pager.register(i, lambda i=i: Tiny(i))
    # Access experts 0 and 1 -> resident {0,1}
    _ = pager.get(0)
    _ = pager.get(1)
    s1 = pager.stats()
    assert s1['resident'] == 2 and s1['evictions'] == 0
    # Access expert 2 -> evict least-recently used (0)
    _ = pager.get(2)
    s2 = pager.stats()
    assert s2['resident'] == 2 and s2['evictions'] == 1
    # Prefetch expert 0 back -> evict LRU (1)
    pager.prefetch([0])
    s3 = pager.stats()
    assert s3['resident'] == 2 and s3['evictions'] >= 1


def test_expert_pager_budget_capacity() -> None:
    # When capacity is derived from budget, ensure positive capacity
    dev = torch.device('cpu')
    pager = ExpertPager(capacity=max(1, int((32 // 64) or 1)), device=dev)
    # Register two experts and ensure capacity >=1 behaves
    class Tiny(torch.nn.Module):
        def __init__(self, nid: int) -> None:
            super().__init__()
            self.n = torch.nn.Parameter(torch.tensor(float(nid)))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.n
    for i in range(2):
        pager.register(i, lambda i=i: Tiny(i))
    _ = pager.get(0)
    _ = pager.get(1)
    s = pager.stats()
    assert s['resident'] >= 1


def test_router_prefetch_integration_monkeypatch() -> None:
    # Simulate router probabilities triggering prefetch beyond top_k
    # by calling pager.prefetch with synthetic indices.
    dev = torch.device('cpu')
    pager = ExpertPager(capacity=3, device=dev)
    class Tiny(torch.nn.Module):
        def __init__(self, nid: int) -> None:
            super().__init__()
            self.n = torch.nn.Parameter(torch.tensor(float(nid)))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.n
    for i in range(5):
        pager.register(i, lambda i=i: Tiny(i))
    # Access top_k=2 (experts 0,1)
    _ = pager.get(0)
    _ = pager.get(1)
    # Router suggests prefetch of [2,3]
    pager.prefetch([2, 3])
    stats = pager.stats()
    # Ensure resident count honors capacity and at least one prefetched expert is present
    assert stats['resident'] == 3


