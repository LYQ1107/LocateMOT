from locatemot.groundjev.trackjev.candidate_competition import (
    competition_rows,entity_clusters,runtime_competition_boxes,sparse_edges,
)


def test_separated_objects_do_not_become_overlapping_competitors():
    # Runtime coordinates describe two disjoint 10x20 objects, at x=100/130.
    boxes=[[100.,100.,110.,120.],[130.,100.,140.,120.]]
    corrected=runtime_competition_boxes(boxes,'xyxy')
    edges=sparse_edges(corrected)
    assert edges==[]
    rows=competition_rows(corrected,entity_clusters(corrected,edges),edges)
    assert rows['competition_iou_rows_milli']==['1000,0','0,1000']
    assert rows['competition_groups']==[[0],[1]]
    # The historical transport made both objects much larger and overlapping.
    assert sparse_edges(runtime_competition_boxes(boxes,'legacy_xywh'))
    assert boxes==[[100.,100.,110.,120.],[130.,100.,140.,120.]]


def test_memory_subset_aligns_indices_geometry_and_crop_evidence():
    from types import SimpleNamespace
    from PIL import Image
    from locatemot.groundjev.trackjev.native_onejev_adapter import NativeOneJevAdapter

    class CaptureRequest:
        def decide(self, request, media):
            self.request, self.media = request, media
            return SimpleNamespace(answers={
                'memory_0': SimpleNamespace(noul=.9),
                'memory_1': SimpleNamespace(noul=.1),
            }), {}

    engine = CaptureRequest()
    adapter = NativeOneJevAdapter('.', engine=engine, candidate_evidence='train_current_crops',
                                  competition_geometry='xyxy', memory_transport='local_xyxy')
    group = [{'box': box} for box in ([0,0,10,20], [100,100,110,120],
                                    [10,0,20,20], [130,100,140,120])]
    result = adapter.decide_memory(image=Image.new('RGB', (160,160)), group=group,
                                  assigned={1:'10',3:'11'}, tracks=[], timestamp=0.,
                                  specification={'task':'mot'}, competition={})
    state = engine.request.state
    assert state['assigned_identity_per_candidate'] == {'0':'10','1':'11'}
    assert state['original_candidate_indices'] == [1,3]
    assert [candidate['index'] for candidate in state['candidates']] == [0,1]
    assert state['competition_iou_rows_milli'] == ['1000,0','0,1000']
    assert state['current_candidate_crop_sheet'] == '<image:2>'
    assert len(engine.media) == 2
    assert engine.media[1]['image'].size == (640,192)
    assert result['updates'] == {1:{'noul':.9},3:{'noul':.1}}


def test_train_birth_question_keeps_native_reject_and_only_applies_to_empty_history():
    from types import SimpleNamespace
    from PIL import Image
    from locatemot.groundjev.trackjev.native_onejev_adapter import NativeOneJevAdapter

    class RejectEngine:
        def decide(self, request, media):
            self.request = request
            return SimpleNamespace(answers={'identity_0': SimpleNamespace(
                probabilities={'NEW':.01,'NO_MATCH':.01,'REJECT':.98})}), {}

    engine = RejectEngine()
    adapter = NativeOneJevAdapter('.', engine=engine, empty_history_question='train_birth_v1')
    arguments = dict(image=Image.new('RGB',(20,20)),group=[{'box':[0,0,10,10]}],
                     timestamp=0.,specification={'task':'mot'})
    output = adapter.decide(tracks=[],**arguments)
    assert output.decisions[0].decision == 'REJECT'
    assert output.decisions[0].probabilities == {'NEW':.01,'NO_MATCH':.01,'REJECT':.98}
    question = engine.request.questions['identity_0']
    assert question.instructions == ('No persistent identity exists yet in this sequence. '
                                     'Which option owns current candidate?')
    assert set(question.criteria) == {'NEW','NO_MATCH','REJECT'}
    request,_ = adapter.build_request(tracks=[{'track_id':10}],**arguments)
    assert request.questions['identity_0'].instructions.startswith('Which identity owns current candidate 0?')


def test_history_images_precede_current_observation_and_are_not_serialized_as_state():
    import json
    from types import SimpleNamespace
    from PIL import Image
    from locatemot.groundjev.trackjev.native_onejev_adapter import NativeOneJevAdapter
    from locatemot.groundjev.trackjev.tracker_decision import TrackJEVTrackerDecision
    from locatemot.groundjev.tracker import OnlineTracker

    class CaptureEngine:
        def __init__(self):self.calls=[]
        def decide(self,request,media):
            self.calls.append((request,media))
            if 'identity_0' in request.questions:
                known=bool(request.state['historical_tracks'])
                probabilities={'1':1.,'NEW':0.,'NO_MATCH':0.,'REJECT':0.} if known else {'NEW':1.,'NO_MATCH':0.,'REJECT':0.}
                answers={'identity_0':SimpleNamespace(probabilities=probabilities)}
            else:answers={'memory_0':SimpleNamespace(noul=.9)}
            return SimpleNamespace(answers=answers),{}

    engine=CaptureEngine()
    adapter=NativeOneJevAdapter('.',engine=engine,candidate_evidence='train_current_crops',
                               memory_transport='local_xyxy',history_evidence='causal_crops')
    tracker=OnlineTracker(TrackJEVTrackerDecision(adapter),no_match_policy='only_if_omitted')
    candidates=[{'box':[0,0,10,10],'label':'person','native_score':None}]
    tracker.step(Image.new('RGB',(20,20),'red'),candidates,0.,{'task':'mot'})
    tracker.step(Image.new('RGB',(20,20),'green'),candidates,1.,{'task':'mot'})
    request,media=engine.calls[1]
    assert request.state['historical_anchor_and_recent_crops']=='<image:3>'
    assert len(media)==3
    assert media[0]['image'].getpixel((5,5))==(0,128,0)
    assert media[2]['image'].getpixel((5,24))==(255,0,0)
    assert tracker.tracks[1].recent_image.getpixel((5,5))==(0,128,0)
    assert 'causal_images' not in json.dumps(request.state)
