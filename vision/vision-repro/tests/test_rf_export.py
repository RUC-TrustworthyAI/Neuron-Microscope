import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from vision_repro.cli import main
from vision_repro.receptive_field_crops import Field, crop_box, model_geometry, sampled_heatmap
from vision_repro.site_export import export, write
from vision_repro.compress_rf_site_snapshot import main as compress

SOURCE = Path(__file__).parent/'fixtures/inceptionv1_widths.py'


def test_geometry_and_boundary_coordinates():
    geometry = model_geometry(SOURCE)
    assert sum(map(len, geometry.values())) == 5804
    assert geometry['conv2d0'][0] == Field(7, 2, 1.5)
    assert geometry['conv2d2'][0] == Field(19, 4, 3.5)
    assert [geometry['mixed3a'][c].size for c in (0, 64, 192)] == [27, 43, 59]
    assert geometry['mixed4c'][447] == Field(299, 16, 15.5)
    assert geometry['mixed5b'][0] == Field(587, 32, 31.5)
    assert crop_box(geometry['conv2d0'][0], (0, 0)) == ((-2., -2., 5., 5.), (0, 0, 7, 7))
    assert crop_box(geometry['conv2d0'][0], (111, 111))[1] == (217, 217, 224, 224)
    assert crop_box(geometry['mixed5b'][0], (3, 3))[1] == (0, 0, 224, 224)
    # Exact cell centres on a linear response ramp; no grid-size approximation.
    ramp = np.arange(4, dtype=float)[None, :] + np.arange(4)[:, None]*10
    np.testing.assert_allclose(sampled_heatmap(ramp, Field(1, 1, .5), (0, 0, 4, 4), 4), ramp)


@pytest.fixture
def snapshot(tmp_path):
    scan = tmp_path/'scan'
    scan.mkdir()
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(224)[None, :]
    pixels[:, :, 1] = np.arange(224)[:, None]
    np.save(scan/'inputs.npy', pixels[None])
    protocol = {'model': {'weights_sha256': 'test', 'path': '/private/model'}}
    write(scan/'protocol.json', protocol)
    manifest = []
    atlas = scan/'web/atlases/00000.webp'
    atlas.parent.mkdir(parents=True)
    Image.new('RGB', (1024, 128), 'gray').save(atlas, lossless=True)
    manifest.append({'path': 'atlases/00000.webp', 'sha256': hashlib.sha256(atlas.read_bytes()).hexdigest()})
    for layer in ('conv2d0', 'mixed5b'):
        raw = np.arange(4)[None, :] + np.arange(4)[:, None] - 3.
        np.save(scan/f'{layer}_maps.npy', np.tile(raw, (1, 48, 1, 1)))
        samples = []
        for i in range(48):
            negative = i >= 32
            samples.append({'index': 0, 'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                'pole': 'negative' if negative else 'positive', 'mapLocation': [0, 0] if negative else [3, 3],
                'activation': -3. if negative else 3., 'atlas': '00000', 'tile': i % 8})
        buckets = [{'id': label, 'examples': samples[a:b]} for label, a, b in
                   [('top', 0, 16), ('p9998', 16, 24), ('p998', 24, 32), ('negative', 32, 48)]]
        node = scan/f'web/neurons/{layer}/0.json'
        node.parent.mkdir(parents=True)
        write(node, {'layer': layer, 'channel': 0, 'exampleBuckets': buckets})
        manifest.append({'path': f'neurons/{layer}/0.json', 'sha256': hashlib.sha256(node.read_bytes()).hexdigest()})
    write(scan/'web/manifest.json', manifest)
    write(scan/'web/complete.json', {'images_scanned': 1,
          'manifest_sha256': hashlib.sha256((scan/'web/manifest.json').read_bytes()).hexdigest()})
    write(scan/'web/protocol.json', protocol)
    output = tmp_path/'rf_test'
    main(['crop-examples', '--scan', str(scan), '--model-source', str(SOURCE), '--output', str(output),
          '--targets', 'conv2d0:0', 'mixed5b:0'])
    return scan, output


def test_rf_all_buckets_and_static_export(snapshot, tmp_path):
    scan, root = snapshot
    output = tmp_path/'site'
    export(output, sample_run=scan, rf_snapshot=root)
    data = json.loads((output/'data.json').read_text())
    assert data['rfExamples']['samples'] == 96
    assert data['publishedApi'] is False
    for layer, side in [('conv2d0', 7), ('mixed5b', 224)]:
        node = json.loads((output/f'rf/nodes/{layer}/0.json').read_text())
        assert len(base64.b64decode(node['heatBase64'])) == 48*24*24
        for bucket in node['exampleBuckets']:
            for sample in bucket['examples']:
                x0, y0, x1, y1 = sample['cropBounds']
                assert (x1-x0, y1-y0) == (side, side)
                assert all(0 <= v <= 1 for v in sample['activationPosition'])
                assert (output/sample['atlasUrl']).is_file()
        assert node['exampleBuckets'][-1]['examples'][0]['activation'] == -3.
    assert (scan/'web/manifest.json').read_bytes() == (output/'samples/manifest.json').read_bytes()
    assert json.loads((scan/'web/neurons/conv2d0/0.json').read_text()).get('cropPolicy') is None


def test_compression_preserves_source_and_alignment(snapshot, tmp_path):
    scan, root = snapshot
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    compressed = tmp_path/'rf_preview'
    compress([str(root), str(compressed)])
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    export(tmp_path/'site', sample_run=scan, rf_snapshot=compressed, published_api=True)
    for layer in ('conv2d0', 'mixed5b'):
        old = json.loads((root/f'examples/rf_test/nodes/{layer}/0.json').read_text())
        new = json.loads((compressed/f'examples/rf_preview/nodes/{layer}/0.json').read_text())
        assert old['heatBase64'] == new['heatBase64']
        assert old['exampleBuckets'][0]['examples'][0]['activationPosition'] == new['exampleBuckets'][0]['examples'][0]['activationPosition']


def test_rf_corruption_rejected(snapshot, tmp_path):
    scan, root = snapshot
    next((root/'images').glob('*.webp')).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        export(tmp_path/'site', sample_run=scan, rf_snapshot=root)


def test_rf_requires_original_sample_scan(tmp_path):
    with pytest.raises(ValueError, match='sample'):
        export(tmp_path/'site', rf_snapshot=tmp_path/'absent')
