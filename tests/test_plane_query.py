import threading
import numpy as np
import pytest
from labelCloud.model.screen_projection import ScreenProjectionCache


def test_pick_foreground_hidden_and_original_index():
    points=np.array([[0,0,.5],[0,0,-.5],[.5,.5,0],[0,0,2]],float)
    cache=ScreenProjectionCache();cache.CHUNK=1
    cache.project(points,np.eye(4),np.eye(4),100,100)
    hidden=np.zeros(4,bool)
    assert cache.pick((50,50),hidden)==1
    hidden[1]=True
    assert cache.pick((50,50),hidden)==0
    hidden[0]=True
    assert cache.pick((50,50),hidden) is None
    assert cache.pick((75,25),hidden)==2
    assert cache.pick((0,0),hidden) is None
    cancel=threading.Event();cancel.set()
    with pytest.raises(InterruptedError): cache.pick((50,50),hidden,cancel=cancel)
