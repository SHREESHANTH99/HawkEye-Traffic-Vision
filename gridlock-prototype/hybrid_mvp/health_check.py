import requests
urls=['http://127.0.0.1:8080/health','http://127.0.0.1:8001/health']
for url in urls:
    try:
        r=requests.get(url,timeout=3)
        print(url,'->',r.status_code,r.text[:200])
    except Exception as e:
        print(url,'-> ERROR',e)
