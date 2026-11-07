#!/usr/bin/env python3
"""Local inference API for FloodGuard AI dashboard."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, make_response, send_from_directory
ROOT=Path(__file__).resolve().parent; MODEL_DIR=ROOT/'model'; MODEL_PATH=MODEL_DIR/'floodguard_rf.joblib'; META_PATH=MODEL_DIR/'metadata.json'
app=Flask(__name__)
DASHBOARD_PATH=ROOT/'FloodGuard_AI_India_Map_ML.html'
FEATURES=['latitude','longitude','elevation_m','month','rain_1d_mm','rain_3d_mm','rain_7d_mm','rain_14d_mm']

def cors(resp):
    resp.headers['Access-Control-Allow-Origin']='*'; resp.headers['Access-Control-Allow-Headers']='Content-Type'; resp.headers['Access-Control-Allow-Methods']='GET,POST,OPTIONS'; return resp
@app.after_request
def add_cors(resp): return cors(resp)
@app.route('/',methods=['GET'])
def dashboard():
    return send_from_directory(ROOT, 'FloodGuard_AI_India_Map_ML.html')


@app.route('/assistant',methods=['POST','OPTIONS'])
def assistant():
    if request.method=='OPTIONS': return cors(make_response('',204))
    data=request.get_json(silent=True) or {}
    question=str(data.get('question','')).strip()[:1200]
    if not question: return jsonify({'status':'bad_request','message':'Question is required.'}),400
    context=data.get('context',{})
    # Optional generative AI: configure OPENAI_API_KEY as a server-side environment secret.
    # Without it, the browser uses its built-in flood knowledge assistant.
    import os, urllib.request, urllib.error
    api_key=os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return jsonify({'status':'fallback','mode':'local','message':'No AI provider configured; use the dashboard local flood assistant.'}),503
    payload={
      'model':os.environ.get('OPENAI_MODEL','gpt-4.1-mini'),
      'instructions':('You are FloodGuard AI Assistant, an educational flood-information helper. Answer flood-related questions clearly and practically. Use the supplied dashboard context only as current displayed context, distinguish forecasts from observations, never invent readings or claim a flood is confirmed, and do not give evacuation decisions as a substitute for authorities. For immediate danger tell the user to follow official local emergency services and disaster-management advisories. If asked about unrelated topics, politely steer back to flood-related information. Keep answers concise but useful.'),
      'input':'User question: '+question+'\nDashboard context (may contain unavailable fields): '+json.dumps(context,ensure_ascii=False)[:4000],
      'max_output_tokens':650
    }
    req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+api_key,'Content-Type':'application/json'},method='POST')
    try:
      with urllib.request.urlopen(req,timeout=18) as r: result=json.loads(r.read().decode())
      answer=result.get('output_text','').strip()
      if not answer:
        for item in result.get('output',[]):
          for part in item.get('content',[]):
            if part.get('type')=='output_text': answer+=part.get('text','')
      if not answer: raise ValueError('AI provider returned no text')
      return jsonify({'status':'ok','mode':'ai','answer':answer})
    except Exception as e:
      # Do not expose keys or internal provider details.
      return jsonify({'status':'provider_unavailable','mode':'local','message':'AI provider unavailable; use local flood knowledge response.'}),503

@app.route('/health',methods=['GET','OPTIONS'])
def health():
    if request.method=='OPTIONS': return cors(make_response('',204))
    if not MODEL_PATH.exists() or not META_PATH.exists(): return jsonify({'status':'model_not_trained','trained':False,'message':'No trained model file found. Run train_model.py after preparing verified village labels.'}),200
    meta=json.loads(META_PATH.read_text(encoding='utf-8')); return jsonify({'status':'ready','trained':True,'target':meta.get('metrics',{}).get('target'),'metrics':meta.get('metrics',{})})
@app.route('/predict',methods=['POST','OPTIONS'])
def predict():
    if request.method=='OPTIONS': return cors(make_response('',204))
    if not MODEL_PATH.exists() or not META_PATH.exists(): return jsonify({'status':'model_not_trained','message':'No trained model exists. No prediction was generated.'}),409
    data=request.get_json(silent=True) or {}
    try:
        row={k:float(data[k]) for k in FEATURES}
        if not all(np.isfinite(v) for v in row.values()): raise ValueError('All features must be finite numbers.')
        if not (-90<=row['latitude']<=90 and -180<=row['longitude']<=180): raise ValueError('Invalid coordinates.')
        model=joblib.load(MODEL_PATH); meta=json.loads(META_PATH.read_text(encoding='utf-8'))
        probability=float(model.predict_proba(pd.DataFrame([row],columns=FEATURES))[0][list(model.named_steps['rf'].classes_).index(1)])
        return jsonify({'status':'ok','trained':True,'probability':probability,'target':meta['metrics']['target'],'metrics':meta['metrics'],'note':'Research prototype only; not an official warning.'})
    except KeyError as e: return jsonify({'status':'bad_request','message':f'Missing feature: {e}'}),400
    except Exception as e: return jsonify({'status':'error','message':str(e)}),400
if __name__=='__main__':
    port=int(__import__('os').environ.get('PORT','8000'))
    host=__import__('os').environ.get('HOST','127.0.0.1')
    print(f'FloodGuard dashboard/API at http://{host}:{port}; press Ctrl+C to stop.')
    app.run(host=host,port=port,debug=False)
