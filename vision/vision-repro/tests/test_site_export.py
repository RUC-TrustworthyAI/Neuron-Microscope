import hashlib
import json

import pytest

from vision_repro.catalog import LAYERS, validate_target
from vision_repro.cli import main
from vision_repro.site_export import export, public_metadata, write


def test_catalog_without_assets(capsys):
    main(['catalog'])
    value = json.loads(capsys.readouterr().out)
    assert value['neurons'] == sum(c for _, c, _ in LAYERS) == 5804
    assert len(value['layers']) == 12


@pytest.mark.parametrize('layer,channels', [('unknown', [0]), ('mixed4c', [-1]),
                                           ('mixed4c', [512]), ('mixed4c', [1, 1]), ('mixed4c', [])])
def test_invalid_target(layer, channels):
    with pytest.raises(ValueError):
        validate_target(layer, channels)


def make_batch(root):
    root.mkdir()
    (root/'items/0').mkdir(parents=True)
    from PIL import Image
    for label in ['initial', 'checkpoint_4', 'final']:
        Image.new('RGB', (4, 4), (120, 130, 140)).save(root/f'items/0/{label}.png')
    write(root/'metadata.json', {'layer':'mixed4c', 'slot':'positive', 'channels':[447],
        'config':{'steps':8, 'size':224, 'seed':7}, 'model':{'weights_sha256':'test'},
        'activation':'post-ReLU', 'direction':'maximize', 'members':1,
        'signed_mapping':[{'node':'test_pre', 'channel':0}]})
    write(root/'complete.json', {'executed_steps':8, 'steps':8})
    write(root/'evaluation.json', {phase:[{'score':[value]}] for phase, value in
                                  [('initial', 0), ('4', 2), ('final', 4)]})
    return root


def test_placeholder_export(tmp_path):
    output = tmp_path/'site'
    result = export(output)
    assert result['neurons'] == 5804 and result['populated_neurons'] == 0
    assert json.loads((output/'our-results.json').read_text()) == {}
    assert json.loads((output/'data.json').read_text())['sampleRoot'] is None
    assert len(json.loads((output/'reference-catalog.json').read_text())['neurons']) == 5804
    with pytest.raises(FileExistsError):
        export(output)


def test_completed_run_actual_steps_and_metadata(tmp_path):
    batch = make_batch(tmp_path/'batch')
    output = tmp_path/'site'
    result = export(output, [batch])
    assert result['exported_variants'] == 1
    node = json.loads((output/'our-results.json').read_text())['mixed4c:447']
    assert set(node['ours']['positive']) == {'0', '4', '8'}
    assert node['variantInfo']['positive']['config']['seed'] == 7
    assert node['variantInfo']['positive']['scores']['8'] == [4]
    for item in json.loads((output/'asset-manifest.json').read_text()):
        assert hashlib.sha256((output/item['path']).read_bytes()).hexdigest() == item['sha256']


def test_duplicate_run_refused(tmp_path):
    batch = make_batch(tmp_path/'batch')
    with pytest.raises(ValueError, match='Duplicate'):
        export(tmp_path/'site', [batch, batch])


def make_samples(root, path='heat/mixed4c/447.bin'):
    root.mkdir()
    payload = b'\x00\x40\x80\xff'
    source = root/path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    manifest = [{'path':path, 'sha256':hashlib.sha256(payload).hexdigest()}]
    # Deliberately compact: export must preserve bytes for the completion digest.
    (root/'manifest.json').write_text(json.dumps(manifest, separators=(',', ':')))
    write(root/'complete.json', {'images_scanned':50000,
        'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()})
    write(root/'protocol.json', {'full_imagenet_scan':False})
    return root


def test_sample_manifest_preserved(tmp_path):
    source = make_samples(tmp_path/'samples')
    output = tmp_path/'site'
    export(output, sample_run=source)
    completion = json.loads((output/'samples/complete.json').read_text())
    assert completion['manifest_sha256'] == hashlib.sha256((output/'samples/manifest.json').read_bytes()).hexdigest()
    assert (output/'samples/heat/mixed4c/447.bin').read_bytes() == b'\x00\x40\x80\xff'


def test_sample_corruption_rejected(tmp_path):
    source = make_samples(tmp_path/'samples')
    (source/'heat/mixed4c/447.bin').write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum'):
        export(tmp_path/'site', sample_run=source)


def test_sample_path_traversal_rejected(tmp_path):
    source = make_samples(tmp_path/'samples', '../outside.bin')
    with pytest.raises(ValueError, match='escapes'):
        export(tmp_path/'site', sample_run=source)


def test_private_path_reduction():
    assert public_metadata({'path':'/home/example/model', 'url':'https://example.com/a'}) == {
        'path':'model', 'url':'https://example.com/a'}


def test_optimize_invalid_cli_before_cuda():
    with pytest.raises(SystemExit):
        main(['optimize', '--layer','mixed4c','--neurons','447','--variant','path_positive',
              '--output','unused'])
