"""全部为协议/mock 测试，不表示真实 SAM 权重推理已验证。"""
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from labelCloud.model.segmentation.base_backend import BaseSegmentationBackend, Prediction
from labelCloud.model.segmentation.patch import SAMPatch, PromptHistory
from labelCloud.model.segmentation.engine import SegmentationEngine
from labelCloud.model.segmentation.sam2_backend import SAMConfig, choose_device, SAM2Backend


class MockBackend(BaseSegmentationBackend):
    device = "cpu"
    def __init__(self, config):
        super().__init__();self.encodes = 0;self.decodes = 0;self.clears = 0
    def set_image(self, image): self.encodes += 1;self.shape = image.shape[:2]
    def predict(self):
        self.decodes += 1
        return Prediction(np.ones((3, *self.shape), bool), np.array([.2, .9, .3]))
    def clear_image(self): self.clears += 1;self.reset_prompts()


class MockPhoto:
    def __init__(self): self.reads = 0
    def read_window(self, bounds, max_size):
        self.reads += 1
        x1, y1, x2, y2 = bounds
        assert max_size <= 2048
        return np.zeros((y2-y1, x2-x1, 4), np.uint8), bounds


@pytest.mark.parametrize('point,expected', [((0,0),(0,0,512,512)),((1999,0),(1487,0,2000,512)),((0,1999),(0,1487,512,2000)),((1999,1999),(1487,1487,2000,2000)),((1000,1000),(488,488,1512,1512))])
def test_patch_clips_four_edges(point, expected):
    patch = SAMPatch.around(point,2000,2000)
    assert patch.bounds == expected
    np.testing.assert_allclose(patch.global_pixel(patch.local(point)),point)


def test_small_image_and_bad_patch():
    assert SAMPatch.around((5,5),10,20).bounds==(0,0,10,20)
    for point in ((-1,0),(10,0),(0,20)):
        with pytest.raises(ValueError): SAMPatch.around(point,10,20)
    with pytest.raises(ValueError): SAMPatch.around((1,1),10,20,4096)


def test_prompt_global_to_local_and_box_undo():
    patch=SAMPatch(100,200,20,30);backend=MockBackend(None);prompts=PromptHistory()
    prompts.events=[(1,(101,202)),(0,(103,204)),('box',(100,200,120,230))]
    prompts.apply(backend,patch)
    assert backend.points==[((1.,2.),1),((3.,4.),0)]
    assert backend.box==(0.,0.,20.,30.)
    prompts.events.append(('box',(102,202,110,210)));prompts.undo()
    assert prompts.box==(100,200,120,230)
    prompts.undo();assert prompts.box is None
    prompts.clear();assert not prompts.events
    with pytest.raises(ValueError): patch.local_box((99,200,110,220))
    with pytest.raises(ValueError): patch.local_box((105,200,105,220))


def test_embedding_reuse_invalidation_model_reuse():
    engine=SegmentationEngine(MockBackend);photo=MockPhoto();patch=SAMPatch(10,20,32,16);prompts=PromptHistory()
    prompts.events=[(1,(12,22))];config=SAMConfig()
    assert engine.run(config,photo,patch,prompts).best()==1
    backend=engine.backend
    prompts.events.append((0,(15,25)));engine.run(config,photo,patch,prompts)
    prompts.clear();engine.run(config,photo,patch,prompts)
    assert backend.encodes==1 and backend.decodes==2 and photo.reads==1
    prompts.events=[(1,(12,22))];engine.run(config,photo,patch,prompts)
    assert backend.encodes==1
    newpatch=SAMPatch(11,20,32,16);engine.run(config,photo,newpatch,prompts)
    assert backend.encodes==2 and engine.backend is backend
    assert patch.image is None and not patch.encoded
    engine.run(config,MockPhoto(),newpatch,prompts)
    assert backend.encodes==3
    engine.clear_image();assert engine.patch_key is None and newpatch.image is None


@pytest.mark.parametrize('available,requested,expected',[(False,'auto','cpu'),(False,'cuda','cpu'),(True,'auto','cuda'),(True,'cpu','cpu')])
def test_device_selection(available,requested,expected):
    torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:available))
    assert choose_device(torch,requested)==expected


def test_optional_imports_no_torch():
    result=subprocess.run([sys.executable,'-c',"import sys; from labelCloud.control.sam_controller import SAMController; assert 'torch' not in sys.modules; assert 'sam2' not in sys.modules"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_missing_dependencies_and_checkpoint(monkeypatch):
    import labelCloud.model.segmentation.sam2_backend as module
    monkeypatch.setattr(module,'availability',lambda:(False,'SAM components missing'))
    with pytest.raises(RuntimeError,match='missing'): SAM2Backend(SAMConfig())
    monkeypatch.setattr(module,'availability',lambda:(True,''))
    with pytest.raises(ValueError,match='checkpoint'): SAM2Backend(SAMConfig(checkpoint='missing-weights-file.pt'))


def test_pixel_mask_reuses_p0_mapper_multi_points():
    pytest.importorskip('pyproj');pytest.importorskip('rasterio')
    from affine import Affine
    from labelCloud.model.coordinate_mapper import CoordinateMapper
    patch=SAMPatch(2,3,2,2);mask=patch.pixel_mask([[True,False],[False,False]])
    assert (mask.col,mask.row)==(2,3)
    mapper=CoordinateMapper(Affine.identity(),10,10,'EPSG:3857','EPSG:3857')
    mapper.build(4,[(np.array([2.1,2.8,3.2,20]),np.array([3.1,3.7,3.2,20]))])
    np.testing.assert_array_equal(mapper.pixel_mask_to_candidate_point_ids(mask),[0,1])
    with pytest.raises(ValueError): patch.pixel_mask(np.ones((3,3)))


def test_retry_after_model_load_error():
    calls=[]
    def factory(config):
        calls.append(1)
        if len(calls)==1: raise RuntimeError('load failed')
        return MockBackend(config)
    engine=SegmentationEngine(factory)
    with pytest.raises(RuntimeError): engine.run(SAMConfig(),None,None,PromptHistory())
    engine.run(SAMConfig(),None,None,PromptHistory())
    assert len(calls)==2


def test_sam_adapter_api_contract_with_fake_modules(monkeypatch,tmp_path):
    """校验适配器参数，不使用也不冒充真实 torch/SAM。"""
    from contextlib import nullcontext
    import types
    import labelCloud.model.segmentation.sam2_backend as module
    checkpoint=tmp_path/'mock.pt';checkpoint.touch()
    calls={}
    class Predictor:
        def __init__(self,model): calls['model']=model
        def set_image(self,image): calls['image']=image
        def predict(self,**kwargs):
            calls['predict']=kwargs
            return np.zeros((3,4,5),bool),np.array([.1,.8,.2]),None
        def reset_predictor(self): calls['reset']=True
    torch=types.ModuleType('torch');torch.cuda=SimpleNamespace(is_available=lambda:False)
    torch.inference_mode=nullcontext
    build=types.ModuleType('sam2.build_sam')
    def build_sam2(config,path,device):
        calls['build']=(config,path,device);return 'mock model'
    build.build_sam2=build_sam2
    predictor=types.ModuleType('sam2.sam2_image_predictor');predictor.SAM2ImagePredictor=Predictor
    for name,value in [('torch',torch),('sam2',types.ModuleType('sam2')),('sam2.build_sam',build),('sam2.sam2_image_predictor',predictor)]:
        monkeypatch.setitem(sys.modules,name,value)
    monkeypatch.setattr(module,'availability',lambda:(True,''))
    backend=SAM2Backend(SAMConfig(checkpoint=str(checkpoint),device='cuda'))
    assert backend.device=='cpu' and calls['build'][2]=='cpu'
    backend.set_image(np.zeros((4,5,3),np.uint8))
    backend.add_positive_point((1,2));backend.add_negative_point((3,2));backend.set_box((0,0,5,4))
    assert backend.predict().best()==1
    np.testing.assert_array_equal(calls['predict']['point_labels'],[1,0])
    np.testing.assert_array_equal(calls['predict']['point_coords'],[[1,2],[3,2]])
    np.testing.assert_array_equal(calls['predict']['box'],[0,0,5,4])
    assert calls['predict']['multimask_output'] is True
    backend.clear_image();assert calls['reset'] and not backend.points


def test_patch_read_failure_is_recoverable():
    engine=SegmentationEngine(MockBackend);prompts=PromptHistory();prompts.events=[(1,(0,0))]
    photo=MockPhoto();patch=SAMPatch(0,0,4,4)
    class BadPhoto:
        def read_window(self,*args,**kwargs): raise OSError('read failure')
    with pytest.raises(OSError): engine.run(SAMConfig(),BadPhoto(),patch,prompts)
    assert engine.patch_key is None
    assert engine.run(SAMConfig(),photo,patch,prompts).best()==1
