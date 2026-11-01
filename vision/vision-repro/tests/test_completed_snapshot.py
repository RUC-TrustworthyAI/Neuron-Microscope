import hashlib
import io
import json
import math

from PIL import Image
import pytest

from vision_repro import full_feature_campaign as campaign
from vision_repro import prepare_completed_snapshot as export
from vision_repro import repair_feature_campaign as repair
from vision_repro.publish_site_snapshot import verified_existing


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(export.compact(value))


def setup(tmp_path, zero=False, seed=0):
    root = tmp_path/'outputs/run'
    root.mkdir(parents=True)
    cfg = {'steps': 1024, 'seed': seed, 'size': 224}
    put(root/'protocol.json', {'config': cfg, 'source_sha256': {}})
    con = campaign.database(root)
    con.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)',
                ('group', 'mixed4c', 'diversity', '[1]', 'done', None, None, 0))
    con.commit(); con.close()
    folder = root/'batches/group'
    meta = {'channels': [1], 'slot': 'diversity', 'layer': 'mixed4c', 'members': 4,
            'config': {**cfg, 'layer': 'mixed4c', 'slot': 'diversity'}, 'model': {},
            'activation': 'post-ReLU', 'aggregation': 'Gaussian', 'direction': 'maximize',
            'signed_mapping': [{'node': 'mixed4c_1x1_pre_relu_conv', 'channel': 1}],
            'source_hashes': {}, 'versions': {}}
    put(folder/'metadata.json', meta)
    put(folder/'complete.json', {**meta, 'steps': 1024})
    put(folder/'evaluation.json', {'initial': [{'score': [0]*4}],
                                  'final': [{'score': [0 if zero else 1, 2, 3, 4]}]})
    put(folder/'history.json', [])
    (folder/'resume.pt').touch()
    item=folder/'items/0';item.mkdir(parents=True)
    Image.new('RGB', (448,448),'red').save(item/'final.png')
    for j in range(4):Image.new('RGB',(224,224),'red').save(item/f'final_member{j}.png')
    base = tmp_path/'base'
    b = io.BytesIO();Image.new('RGB',(224,224),'blue').save(b,format='WEBP',lossless=True)
    raw=b.getvalue();sha=hashlib.sha256(raw).hexdigest();image=f'images/{sha}.webp'
    (base/'images').mkdir(parents=True);(base/image).write_bytes(raw)
    slots=[s for s in export.SLOTS if s!='diversity']
    node={'layer':'mixed4c','channel':1,'model':{},'config':{},
          'ours':{s:{'1024':f'/media/{sha}.webp'} for s in slots},
          'variantInfo':{s:{'batchSize':1,'image_sha256':sha,'config':{},
                            'final':{'score':[1 if s in export.SLOTS[:3] else -1]}} for s in slots},
          'pending':{'diversity':'quality_review'}}
    nodekey='releases/old/nodes/mixed4c/1.json'
    put(base/nodekey,node);put(base/'releases/old/catalog.json',{'nodes':{'mixed4c:1':'11011'}})
    put(base/'upload-manifest.json',{'summary':{'release':'old'},'objects':[
        {'key':image,'sha256':sha,'bytes':len(raw)},
        {'key':nodekey,'sha256':export.sha((base/nodekey).read_bytes()),'bytes':(base/nodekey).stat().st_size}]})
    return root, base, node


def test_merge_published_modes_with_new_missing_mode(tmp_path):
    root,base,_=setup(tmp_path)
    out=tmp_path/'release'
    export.main([str(root),str(out),'--base-snapshot',str(base),'--complete-neurons-only'])
    result=repair.read(out/'complete.json')
    assert result['fullyValidFiveModeNeurons']==1 and result['variants']==5 and result['images']==8
    node=repair.read(out/'releases/release/nodes/mixed4c/1.json')
    assert set(node['ours'])==set(export.SLOTS) and not node['pending']


def test_incomplete_group_preserves_old_node(tmp_path):
    root,base,old=setup(tmp_path,zero=True)
    out=tmp_path/'release'
    export.main([str(root),str(out),'--base-snapshot',str(base),'--complete-neurons-only'])
    assert repair.read(out/'releases/release/nodes/mixed4c/1.json')==old
    assert repair.read(out/'complete.json')['updatedCompleteNeurons']==0


def test_include_zero_diversity_preserves_members_and_real_scores(tmp_path):
    root,base,_=setup(tmp_path,zero=True)
    out=tmp_path/'release'
    export.main([str(root),str(out),'--base-snapshot',str(base),
                 '--complete-neurons-only','--include-zero-diversity'])
    result=repair.read(out/'complete.json')
    assert result['completeFiveModeNeurons']==1
    assert result['fullyValidFiveModeNeurons']==0
    assert result['zeroActivationDiversityNeurons']==1
    assert result['variants']==5 and result['images']==8
    assert len(result['qualityFlags'])==1
    node=repair.read(out/'releases/release/nodes/mixed4c/1.json')
    assert node['variantInfo']['diversity']['final']['score']==[0,2,3,4]
    assert node['variantInfo']['diversity']['batchSize']==4 and not node['pending']
    raw=(root/'batches/group/items/0/final.png').read_bytes()
    assert node['variantInfo']['diversity']['source_png_sha256']==export.sha(raw)


@pytest.mark.parametrize('slot', ['positive','path_positive','negative','path_negative'])
def test_zero_activation_display_option_retains_single_image_modes(tmp_path,slot):
    root,_,_=setup(tmp_path)
    con=campaign.database(root)
    con.execute('UPDATE jobs SET slot=?',(slot,));con.commit();con.close()
    folder=root/'batches/group'
    for name in ['metadata.json','complete.json']:
        data=repair.read(folder/name)
        data.update(slot=slot,members=1)
        data['config']['slot']=slot
        data['activation']='pre-ReLU' if slot in export.SLOTS[3:] else 'post-ReLU'
        data['direction']='minimize' if slot in export.SLOTS[3:] else 'maximize'
        put(folder/name,data)
    put(folder/'evaluation.json',{'initial':[{'score':[0]}],'final':[{'score':[0]}]})
    Image.new('RGB',(224,224),'red').save(folder/'items/0/final.png')
    strict=tmp_path/'strict';out=tmp_path/'release'
    export.main([str(root),str(strict),'--include-zero-diversity'])
    assert repair.read(strict/'complete.json')['variants']==0
    export.main([str(root),str(out),'--include-zero-activations'])
    node=repair.read(out/'releases/release/nodes/mixed4c/1.json')
    assert set(node['ours'])=={slot}
    assert node['variantInfo'][slot]['final']['score']==[0]
    assert repair.read(out/'complete.json')['zeroActivationPolicy']=='include_all'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1])
def test_zero_diversity_option_does_not_admit_invalid_values(tmp_path,bad):
    root,base,old=setup(tmp_path)
    put(root/'batches/group/evaluation.json', {'initial':[{'score':[0]*4}],
        'final':[{'score':[bad,2,3,4]}]})
    args=[str(root),str(tmp_path/'release'),'--base-snapshot',str(base),
          '--complete-neurons-only','--include-zero-diversity']
    if math.isfinite(bad):
        export.main(args)
        assert repair.read(tmp_path/'release/releases/release/nodes/mixed4c/1.json')==old
    else:
        with pytest.raises(AssertionError):export.main(args)


def test_repaired_seed_is_retained_and_checksum_checked(tmp_path,monkeypatch):
    root,base,_=setup(tmp_path,zero=True)
    seed,_,_=setup(tmp_path/'repair',seed=2)
    row=repair.rows(seed)[0]
    receipt=repair.validate_batch(seed,row)[0]
    repair_root=tmp_path/'repair'
    put(repair_root/'accepted.json',receipt)
    out=tmp_path/'release'
    monkeypatch.chdir(tmp_path)
    export.main([str(root.relative_to(tmp_path)),'release','--base-snapshot','base','--complete-neurons-only',
                 '--repair-root','repair'])
    node=repair.read(out/'releases/release/nodes/mixed4c/1.json')
    assert node['variantInfo']['diversity']['config']['seed']==2
    assert node['variantInfo']['diversity']['final']['score']==[1,2,3,4]
    receipt['mixed4c:1:diversity']['images_sha256']['final.png']='wrong'
    put(repair_root/'accepted.json',receipt)
    with pytest.raises(ValueError, match='receipt differs'):
        export.main([str(root),str(tmp_path/'bad'),'--base-snapshot',str(base),
                     '--complete-neurons-only','--repair-root',str(repair_root)])


def test_cloud_reuse_requires_matching_receipts():
    records=[{'key':str(i),'sha256':'x','bytes':1} for i in range(205)]
    def request(_route,body):
        batch=body['objects']
        return {'checked':len(batch),'bad':[r['key'] for r in batch if int(r['key'])%2]}
    assert len(verified_existing(records,request))==103
    with pytest.raises(ValueError):
        verified_existing(records,lambda *_:{'checked':100,'bad':['unrequested']})
