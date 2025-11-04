from omnicoder.inference.runtimes.kv_prefetch import PagingConfig, LRUKVPager


def test_kv_prefetch_miss_rate_and_stalls():
    cfg = PagingConfig(n_layers=4, heads=8, page_len=16, max_pages_ram=16, prefetch_ahead=1)
    pager = LRUKVPager(cfg)
    seq_len = 256
    for pos in range(seq_len):
        pager.access(pos)
    s = pager.stats()
    assert 0.0 <= s["miss_rate"] <= 1.0
    assert 0.0 <= s["stall_ratio"] <= 1.0
    # With prefetch, hit_rate should be reasonable on sequential access
    assert s["hit_rate"] > 0.5


