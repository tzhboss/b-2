"""Read public source routes; preserve failures and do not assign query truth."""
import json,time,urllib.request,urllib.parse
from pathlib import Path
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parent
class Links(HTMLParser):
 def __init__(self): super().__init__(); self.links=[]
 def handle_starttag(self,tag,attrs):
  if tag=='a':
   for k,v in attrs:
    if k=='href': self.links.append(v)

params=urllib.parse.urlencode({'alt':'json','max-results':'50',
 'published-min':'2014-11-01T00:00:00-05:00','published-max':'2014-11-04T23:59:59-05:00'})
routes=[('peru_public_feed','https://mreperu.blogspot.com/feeds/posts/default?'+params),
 ('apsn_known_digest','https://www.archive.asia-pacific-solidarity.net/southeastasia/indonesia/netnews/2008/ind26v12.htm')]
records=[]
for name,url in routes:
 started=time.perf_counter()
 try:
  request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
  with urllib.request.urlopen(request) as response:
   body=response.read(); record={'route':name,'url':url,'final_url':response.url,
    'status':response.status,'content_type':response.headers.get('Content-Type'),'bytes':len(body)}
  (ROOT/(name+'-round17.response')).write_bytes(body)
  if name=='peru_public_feed':
   data=json.loads(body)
   entries=data.get('feed',{}).get('entry',[])
   posts=[{'title':x.get('title',{}).get('$t'),'published':x.get('published',{}).get('$t'),
    'links':x.get('link',[]),'content':x.get('content',{}).get('$t')} for x in entries]
   (ROOT/'peru-feed-posts-round17.json').write_text(json.dumps(posts,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
   record['posts_n']=len(posts); record['post_titles']=[x['title'] for x in posts]
  else:
   parser=Links();parser.feed(body.decode('utf8',errors='replace'))
   record['nonfragment_links']=[u for u in parser.links if not u.startswith('#')]
 except Exception as exc:
  record={'route':name,'url':url,'error_type':type(exc).__name__,'error':str(exc)}
 record['request_seconds']=time.perf_counter()-started
 records.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
 (ROOT/'source-route-fetches-round17.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
