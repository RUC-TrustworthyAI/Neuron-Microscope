import json

from PIL import Image
import pytest

from vision_repro import full_feature_campaign as campaign
from vision_repro import repair_feature_campaign as repair


def baseline(tmp_path):
    root = tmp_path/'baseline'
    root.mkdir()
    (root/'anchors').mkdir()
    repair.save(root/'protocol.json', {'config': {'size': 2, 'steps': 1024, 'seed': 0,
                                                 'channel': 99, 'layer': 'mixed4c'},
                                      'source_sha256': {}, 'steps': 1024})
    con = campaign.database(root)
    con.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)',
                ('original', 'mixed4c', 'diversity', '[1,3]', 'done', None, None, 0))
    con.commit()
    con.close()
    folder = root/'batches/original'
    folder.mkdir(parents=True)
    repair.save(folder/'evaluation.json', {'final': [{'score': [0., 2., 3., 4.]}, {'score': [1., 2., 3., 4.]}]})
    audit = tmp_path/'audit.json'
    repair.save(audit, {'flagged_variants': [{'layer': 'mixed4c', 'neuron': 1, 'slot': 'diversity',
                                            'scores': [0., 2., 3., 4.]}]})
    return root, audit


def completed_round(tmp_path, scores=(1., 2., 3., 4.)):
    base, audit = baseline(tmp_path)
    root = tmp_path/'round'
    repair.prepare_round(root, base, repair.audited_targets(base, audit), 1, 4)
    row = repair.rows(root)[0]
    folder = root/'batches'/row['id']
    item = folder/'items/0'
    item.mkdir(parents=True)
    cfg = repair.read(root/'protocol.json')['config']
    cfg.pop('channel')
    cfg.update(layer='mixed4c', slot='diversity')
    complete = {'channels': [1], 'layer': 'mixed4c', 'slot': 'diversity', 'members': 4, 'steps': 1024}
    repair.save(folder/'complete.json', complete)
    repair.save(folder/'metadata.json', {**complete, 'activation': 'post-ReLU', 'config': cfg})
    repair.save(folder/'evaluation.json', {'final': [{'score': list(scores)}]})
    repair.save(folder/'history.json', [])
    (folder/'resume.pt').touch()
    composite = Image.new('RGB', (4, 4))
    for j in range(4):
        member = Image.new('RGB', (2, 2), (j*30, 10, 20))
        member.save(item/f'final_member{j}.png')
        composite.paste(member, (j%2*2, j//2*2))
    composite.save(item/'final.png')
    return root, row, folder


def test_sparse_selection_and_resume_preserve_baseline(tmp_path):
    base, audit = baseline(tmp_path)
    original = (base/'protocol.json').read_bytes()
    targets = repair.audited_targets(base, audit)
    root = tmp_path/'round'
    repair.prepare_round(root, base, targets, 1, 4)
    con = campaign.database(root)
    con.execute("UPDATE jobs SET state='done'")
    con.commit()
    con.close()
    repair.prepare_round(root, base, targets, 1, 4)
    assert len(repair.rows(root)) == 1
    assert repair.rows(root)[0]['state'] == 'done'
    assert json.loads(repair.rows(root)[0]['channels']) == [1]
    assert (base/'protocol.json').read_bytes() == original
    assert repair.read(root/'protocol.json')['config']['seed'] == 1
    with pytest.raises(ValueError, match='existing repair round'):
        repair.prepare_round(root, base, targets, 2, 4)


def test_reject_stale_audit_and_live_baseline(tmp_path):
    base, audit = baseline(tmp_path)
    data = repair.read(audit)
    data['flagged_variants'][0]['scores'][1] = 999
    repair.save(audit, data)
    with pytest.raises(ValueError, match='Audit differs'):
        repair.audited_targets(base, audit)
    con = campaign.database(base)
    con.execute("UPDATE jobs SET state='running'")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match='must have finished'):
        repair.audited_targets(base, audit)


def test_accept_complete_group_and_reject_zero_member(tmp_path):
    root, row, folder = completed_round(tmp_path)
    good, bad = repair.validate_batch(root, row)
    assert not bad
    assert good['mixed4c:1:diversity']['seed'] == 1
    assert len(good['mixed4c:1:diversity']['images_sha256']) == 5
    repair.save(folder/'evaluation.json', {'final': [{'score': [1., 0., 3., 4.]}]})
    good, bad = repair.validate_batch(root, row)
    assert not good and 'mixed4c:1:diversity' in bad


@pytest.mark.parametrize('problem', ['nan', 'preview', 'missing', 'config'])
def test_reject_invalid_artifact_or_provenance(tmp_path, problem):
    root, row, folder = completed_round(tmp_path)
    if problem == 'nan':
        repair.save(folder/'evaluation.json', {'final': [{'score': [1., float('nan'), 3., 4.]}]})
    elif problem == 'preview':
        Image.new('RGB', (4, 4), 'white').save(folder/'items/0/final.png')
    elif problem == 'missing':
        (folder/'resume.pt').unlink()
    else:
        meta = repair.read(folder/'metadata.json')
        meta['config']['seed'] = 9
        repair.save(folder/'metadata.json', meta)
    with pytest.raises(ValueError):
        repair.validate_batch(root, row)
