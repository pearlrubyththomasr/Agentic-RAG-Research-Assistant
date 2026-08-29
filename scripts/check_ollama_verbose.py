import os, requests, sys
base=os.getenv('OLLAMA_BASE_URL',os.getenv('OLLAMA_HOST','http://127.0.0.1:11434')).rstrip('/')
url=base+'/v1/models'
print('Checking', url)
try:
    r=requests.get(url, timeout=20)
    print('Status', r.status_code)
    print(r.text[:2000])
except Exception as e:
    print('Error:', e)
    sys.exit(1)
