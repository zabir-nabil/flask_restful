import requests
import jsonschema._format
import json

r = requests.post('http://127.0.0.1:5000/upload', files={'file': open('paper.pdf', 'rb')})
print(r.status_code)
#print(r.text)

rjson = r.json()

rj = json.loads(rjson)

print(type(rj["data"]))

import io, json
with io.open('data.json', 'w', encoding='utf-8') as f:
  f.write(json.dumps(rj, ensure_ascii=False))
