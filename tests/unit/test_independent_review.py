from __future__ import annotations
import json
import tempfile
import threading
import urllib.error
import urllib.request
import unittest
from pathlib import Path
from agentinputlab.annotation.independent import ReviewStudy,FIELDS,sha,handler
from http.server import ThreadingHTTPServer
from agentinputlab.experiments.measurement import MeasurementItem,build_measurement_assignment

class IndependentReviewTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);(self.root/'private').mkdir()
  items=[MeasurementItem('one','g1','contract_anchor',{'query':'Set an alarm at 6:30.'}),MeasurementItem('two','g2','defect_eligibility',{'query':'Fix it.'})]
  assignment=build_measurement_assignment(items,annotator_ids=('R01','R02','R03'),seed=2027,allow_partial=True)
  (self.root/'assignment_manifest.json').write_text(json.dumps(assignment['manifest']))
  for r,p in assignment['packets'].items():(self.root/'private'/f'{r}.json').write_text(json.dumps(p))
  (self.root/'private/reviewer_access.json').write_text(json.dumps({r:{'code_sha256':sha((r+'code').encode())} for r in ('R01','R02','R03')}))
  self.study=ReviewStudy(self.root)
 def tearDown(self):self.temp.cleanup()
 def test_login_requires_secret_and_independence(self):
  with self.assertRaises(ValueError):self.study.login('R01','wrong',True)
  with self.assertRaises(ValueError):self.study.login('R01','R01code',False)
  token=self.study.login('R01','R01code',True);self.assertEqual(self.study.reviewer(token),'R01')
 def test_two_independent_votes_and_immutable_submission(self):
  blind=next(iter(self.study.items));sample=self.study.short[blind];pair=self.study.assignment[blind]
  third=next(r for r in ('R01','R02','R03') if r not in pair)
  with self.assertRaises(ValueError):self.study.item(third,sample)
  labels={k:'yes' for k in FIELDS[self.study.items[blind]['kind']]}
  self.study.record(pair[0],sample,labels,'Original input contains explicit time.',False)
  self.study.record(pair[0],sample,labels,'Original input contains explicit time.',True)
  with self.assertRaises(ValueError):self.study.record(pair[0],sample,labels,'change',False)
  labels={k:'no' for k in labels};self.study.record(pair[1],sample,labels,'Independent interpretation.',True)
  item=self.study.item(third,sample);self.assertEqual(item['stage'],'adjudication');self.assertEqual(item['labels'],{})
  self.assertNotIn('other_labels',item);self.study.record(third,sample,labels,'Independent third review.',True)
 def test_own_answers_only_unknown_fields_rejected(self):
  blind=next(iter(self.study.items));sample=self.study.short[blind];pair=self.study.assignment[blind]
  with self.assertRaises(ValueError):self.study.record(pair[0],sample,{'score':'yes'},'reason',False)
  with self.assertRaises(ValueError):self.study.record(pair[0],sample,{},'reason',True)
  self.assertNotIn('automatic_input_label',self.study.item(pair[0],sample))
 def test_packet_tamper_rejected(self):
  path=self.root/'private/R01.json';packet=json.loads(path.read_text());packet[0]['payload']['query']='tampered';path.write_text(json.dumps(packet))
  with self.assertRaisesRegex(ValueError,'packet changed'):ReviewStudy(self.root)
 def test_missing_real_votes_never_become_completed_human_analysis(self):
  out=self.root/'export.json';result=self.study.export(out);self.assertFalse(result['analysis_complete']);self.assertEqual(result['submitted_votes'],0)
 def test_single_category_agreement_not_reported_as_defined_kappa(self):
  for blind,entry in self.study.items.items():
   labels={k:'yes' for k in FIELDS[entry['kind']]}
   for reviewer in self.study.assignment[blind]:self.study.record(reviewer,self.study.short[blind],labels,'Unit fixture, not actual human evidence.',True)
  path=self.root/'all-agree.json';self.assertTrue(self.study.export(path)['analysis_complete'])
  agreement=json.loads(path.read_text())['analysis']['agreement']
  for field in agreement:
   self.assertEqual(field['percent_agreement'],1)
   self.assertIsNone(field['cohen_kappa']);self.assertIsNone(field['krippendorff_alpha_nominal'])
   self.assertEqual(field['coefficient_status'],'undefined_single_category')
 def test_http_blocks_cross_origin_and_private_files(self):
  server=ThreadingHTTPServer(('127.0.0.1',0),handler(self.study));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  base=f'http://127.0.0.1:{server.server_port}'
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  try:
   for path in ('/private/reviewer_invites.json','/api/status','/api/versions'):
    with self.assertRaises(urllib.error.HTTPError):opener.open(base+path,timeout=3)
   payload=json.dumps({'reviewer_id':'R01','code':'R01code','independent':True}).encode()
   req=urllib.request.Request(base+'/api/login',data=payload,headers={'Content-Type':'application/json','Origin':'https://foreign.invalid'})
   with self.assertRaises(urllib.error.HTTPError):opener.open(req,timeout=3)
   req=urllib.request.Request(base+'/api/login',data=payload,headers={'Content-Type':'application/json','Origin':base})
   result=json.load(opener.open(req,timeout=3));self.assertIn('token',result)
   req=urllib.request.Request(base+'/api/status',headers={'Authorization':'Bearer '+result['token']})
   result=json.load(opener.open(req,timeout=3));self.assertNotIn('secret',result);self.assertNotIn('annotations',result)
  finally:server.shutdown();server.server_close();thread.join()
if __name__=='__main__':unittest.main()
