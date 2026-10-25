from pathlib import Path
import json,sys,re,time,hashlib,unicodedata,concurrent.futures
from urllib.parse import quote
import requests
ROOT=Path(r'C:\desktop\مقاله\مقاله پایان نامه\Revision_20261024');CTRL=ROOT/'00_Revision_Control';OUT=Path(__file__).parent
sys.path.insert(0,str(CTRL));import build_revision as bld
used=set()
for p in ROOT.glob('*/QUALITY_CONTROLS/citation_mapping.json'):used.update(json.loads(p.read_text('utf-8'))['order'])
def norm(s):return re.sub(r'[^a-z0-9]+','',unicodedata.normalize('NFKD',re.sub('<[^>]+>','',s)).casefold())
def run(key):
    x=bld.b.DATA[key];doi=x.get('doi')
    if not doi:return {'key':key,'status':'IDENTIFIED_SOFTWARE_DISCLOSURE','title':x['title'],'source':x.get('url'),'scientific_evidence':False}
    url='https://api.crossref.org/works/'+quote(doi,safe='')
    for attempt in range(2):
        try:
            response=requests.get(url,headers={'User-Agent':'Local-manuscript-reference-audit/2026'},timeout=35);response.raise_for_status();message=response.json()['message'];break
        except Exception as exc:
            if attempt==1:return {'key':key,'doi':doi,'status':'RETRIEVAL_UNRESOLVED','error':type(exc).__name__+': '+str(exc),'url':url}
            time.sleep(1)
    raw=json.dumps(message,ensure_ascii=False,sort_keys=True).encode('utf-8');(OUT/'reference_registry').mkdir(exist_ok=True);(OUT/'reference_registry'/(key+'.json')).write_bytes(raw)
    titles=message.get('title',[]);title=' '.join(titles);subtitle=' '.join(message.get('subtitle',[]));variants=[title,title+': '+subtitle]
    # Author-date capitalization and typographic punctuation do not change source identity.
    exact=any(norm(x['title'])==norm(t) for t in variants);contained=any(norm(x['title']) in norm(t) or norm(t) in norm(x['title']) for t in variants if t)
    authors=[v.get('family',v.get('name','')) for v in message.get('author',[])];notices=[]
    for entry in message.get('update-to',[]):
        if any(w in str(entry).lower() for w in ['retract','withdraw']):notices.append(entry)
    status='IDENTITY_VERIFIED' if str(message.get('DOI','')).casefold()==doi.casefold() and (exact or contained) else 'METADATA_REVIEW'
    result={'key':key,'doi':doi,'url':url,'publisher_url':message.get('URL'),'status':status,'manuscript_title':x['title'],'registered_title':title,'registered_subtitle':subtitle,'title_match':'exact_after_typographic_normalization' if exact else 'title_subtitle_containment' if contained else 'mismatch','manuscript_year':x['year'],'registry_dates':{k:message.get(k) for k in ['published','published-print','published-online','issued'] if k in message},'manuscript_journal':x['journal'],'registered_journal':message.get('container-title'),'registered_authors':authors,'crossref_record_sha256':hashlib.sha256(raw).hexdigest(),'retraction_markers_in_record':notices,'entailment_status':'CHECK_SEPARATELY_FROM_METADATA','status_check_scope':'Crossref record only; absence of a marker is not exhaustive retraction certification.'}
    print(key,status,doi,flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(run,sorted(used,key=lambda k:(k.startswith('J'),int(k.removeprefix('J'))))))
summary={'retrieved_on':'2026-10-24','source':'Crossref official DOI registry API; live retrieval','reference_count':len(results),'scientific_doi_count':sum(bool(x.get('doi')) for x in results),'identity_verified':sum(x['status']=='IDENTITY_VERIFIED' for x in results),'unresolved_or_review':[x for x in results if x['status'] in ['METADATA_REVIEW','RETRIEVAL_UNRESOLVED']],'references':results,'claim_limit':'DOI identity is distinct from full-text support, scientific validity, retraction completeness and journal acceptance.'}
(OUT/'aggregate/REFERENCE_IDENTITY_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
print('REFERENCE_AUDIT_COMPLETE',summary['reference_count'],summary['identity_verified'],'unresolved',len(summary['unresolved_or_review']),flush=True)
