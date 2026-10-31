import json

import pytest

from vision_repro import full_feature_campaign as campaign
from vision_repro.resume_published_campaign import exclude_published


def fixture_queue(tmp_path):
    (tmp_path / 'protocol.json').write_text('{}')
    con = campaign.database(tmp_path)
    for slot in campaign.SLOTS:
        con.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)',
                    ('conv2d0_' + slot, 'conv2d0', slot, '[0,1,2]', 'pending', None, None, 0))
    con.commit()
    con.close()
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps({'release': 'test', 'nodes': {'conv2d0:0': '11111', 'conv2d0:1': '11q11'}}))
    return path


def test_partial_and_quality_flagged_modes_remain_pending(tmp_path):
    path = fixture_queue(tmp_path)
    report = exclude_published(tmp_path, path)
    assert report['published_variants_preserved'] == 9
    assert report['missing_variants'] == 6
    assert report['missing_images'] == 12
    assert report['missing_neurons'] == 2
    con = campaign.database(tmp_path)
    targets = {}
    for row in con.execute("SELECT * FROM jobs WHERE state!='split'"):
        for ch in json.loads(row['channels']):
            key = (ch, row['slot'])
            assert key not in targets
            targets[key] = row['state']
    con.close()
    assert len(targets) == 15
    assert targets[1, 'diversity'] == 'pending'
    assert all(targets[0, slot] == 'published' for slot in campaign.SLOTS)
    assert exclude_published(tmp_path, path) == report


def test_rejects_active_queue_without_altering_it(tmp_path):
    path = fixture_queue(tmp_path)
    con = campaign.database(tmp_path)
    con.execute("UPDATE jobs SET state='running' WHERE slot='positive'")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match='fresh'):
        exclude_published(tmp_path, path)
    con = campaign.database(tmp_path)
    assert con.execute("SELECT COUNT(*) FROM jobs WHERE state='published'").fetchone()[0] == 0
    con.close()


def test_rejects_a_different_published_baseline(tmp_path):
    path = fixture_queue(tmp_path)
    exclude_published(tmp_path, path)
    path.write_text(json.dumps({'release': 'different', 'nodes': {}}))
    with pytest.raises(ValueError, match='baseline'):
        exclude_published(tmp_path, path)
