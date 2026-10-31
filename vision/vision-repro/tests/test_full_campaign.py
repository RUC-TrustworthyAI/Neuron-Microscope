import json

import torch

from vision_repro import full_feature_campaign as campaign
from vision_repro.profile_full_concurrency import projected_memory_mib, stop_owned_children


def test_queue_covers_each_variant_once_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'LAYERS', [('mixed4c', 7, 14)])
    anchor_dir = tmp_path/'anchors'
    anchor_dir.mkdir()
    (anchor_dir/'complete.json').write_text('{}')
    (anchor_dir/'protocol.json').write_text('{}')
    anchors = {f'{ch}_{pole}': {'usable': True} for ch in range(7) for pole in ('positive','negative')}
    anchors['3_negative']['usable'] = False
    torch.save(anchors, anchor_dir/'mixed4c.pt')
    campaign.initialize(tmp_path, 4)
    campaign.initialize(tmp_path, 4)
    con = campaign.database(tmp_path)
    seen = set()
    for row in con.execute('SELECT * FROM jobs'):
        channels = json.loads(row['channels'])
        assert len(channels) <= (1 if row['slot'] == 'diversity' else 4)
        for ch in channels:
            key = (row['slot'], ch)
            assert key not in seen
            seen.add(key)
            assert (row['state'] == 'unavailable') == (key == ('path_negative', 3))
    assert len(seen) == 7*5
    con.close()


def test_unified_memory_estimate_accounts_for_each_context():
    assert projected_memory_mib(2,16) == 6560
    assert projected_memory_mib(4,8) > projected_memory_mib(1,32)


def test_cleanup_terminates_only_owned_live_children():
    class Child:
        def __init__(self, finished):
            self.finished = finished
            self.terminated = False
        def poll(self):
            return 0 if self.finished else None
        def terminate(self):
            self.terminated = True
        def wait(self, timeout):
            self.finished = True
    finished, active = Child(True), Child(False)
    stop_owned_children([finished, active])
    assert not finished.terminated and active.terminated
