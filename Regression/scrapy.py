import requests
import webbrowser
browser=webbrowser.Chrome()
html=requests.get('http://wechat.aixiang160.com/pages/voteDetail/main?actId=233')
html.encoding=html.apparent_encoding
print(html.text)