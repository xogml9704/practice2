
from urllib.parse import urlparse

from matplotlib.pyplot import show


urllib 패키지의 주요 모듈
 urllib
  URL 작업을 위한 여러 모듈을 모은 패키지
  파이썬의 표준 라이브러리

 URL 문자열과 웹 요청에 관련된 모듈 5개 제공

 urllib.request
 -URL 문자열을 가지고 요청 기능 제공

 urllib.response
 -urllib 모듈에 의해 사용되는 응답 클래스들 제공

 urllib.parse
 -URL 문자열을 파싱하여 해석하는 기능 제공

 urllib.error
 -urllib.request에 의해 발생하는 예외 클래스들 제공

 urllib.robotparser
 -robots.txt 파일을 구문 분석하는 기능 제공

 URL 문자열을 가지고 HTTP 요청을 수행하는 urllib.request 모듈
 URL 문자열(주소)을 해석하는 urllib.parse 모듈

 urllib 패키지의 주요 모듈
  urllib.request 모듈
   URL 문자열을 가지고 HTTP 요청을 수행
   urlopen() 함수를 사용하여 웹 서버에 페이지를 요청하고, 서버로부터
   받은 응답을 저장하여 응답 객체(http.client.HTTPResponse)를 반환

 res = urllib.request.urlopen
 ("요청하려는 페이지의 URL 문자열")
 urllib.request.urlopen
 ("URL 문자열")

 urllib 패키지의 주요 모듈
 http.client.HTTPResponse 클래스
  웹 서버로부터 받은 응답을 래핑하는 객체
  응답 헤더나 응답 바디의 내용을 추출하는 메서드 제공
  HTTPResponse.read([amt])
  HTTPResponse.readinto(b)
  HTTPResponse.getheader(name, default=None)
  HTTPResponse.getheaders()
  HTTPResponse.msg
  HTTPResponse.version
  HTTPResponse.status
  HTTPResponse.reason
  HTTPResponse.closed

 urllib 패키지의 주요 모듈
  http.client.HTTPResponse 객체의 read() 메서드
  read() 메서드를 실행하면 웹 서버가 전달한 데이터(응답 바디)를
  바이트열로 읽어 들임
  바이트열
   16진수로 이루어진 수열이기 때문에 읽기 어려우므로 웹 서버가 보낸
   한글을 포함한 텍스트 형식의 HTML 문서의 내용을 읽을 때는 텍스트
   형식으로 변환함
   바이트열(bytes)의 decode('문자 셋')메서드를 실행하여 응답된
   문자 셋에 알맞은 문자로 변환함

 res.read()

 res.read().decode('utf-8')
 <body>
 <h1>가나다ABC</h1>
 </body>

 웹 페이지 인코딩 체크(1)
  웹 크롤링하려는 웹 페이지가 어떠한 문자 셋으로 작성되었는지 파악하는 것이 필수
  페이지의 문자 셋 정보
   페이지의 소스 내용에서 <meta> 태그의 charset 정보를 체크하면 파악 가능

 웹 페이지 인코딩 체크(2)
  웹 페이지의 문자 셋 정보를 파이썬 프로그램으로도 파악할 수 있음
  사용되는 API http.client.HTTPMessage 객체의 get_content_charset() 메서드

  urllib.request.urlopen() 함수의 리턴 값인
  http.client.HTTPResponse 객체의 info() 메서드 호출

  http.client.HTTPMessage 객체가 리턴 됨

  get_content_charset() 메서드 호출

  문자 셋 정보를 문자열로 리턴 받음

  웹 서버로부터 응답될 때 전달되는 Content-Type이라는 응답 헤더 정보를
  읽고 해당 페이지의 문자 셋 정보를 추출해 줌
  url = 'http://www.python.org/'
  f = urllib.request.urlopen(url)
  encoding = f.info().get_content_charset()

 웹 서버에 페이지 또는 정보를 요청할 때
 함께 전달하는 데이터
  GET 방식 요청 : Query 문자열
  POST 방식 요청 : 요청 파라미터
  name = value&name=value&name=value&.....
  영문과 숫자는 그대로 전달되지만 한글은 %기호와 함께 16진수 코드 값으로
  전달되어야 함
  웹 크롤링을 할 때 요구되는 Query 문자열을 함께 전달해야 하는 경우, 직접
  Query 문자열을 구성해서 전달해야 함.

 urllib.parse 모듈 사용
 urllib.parse.urlparse()
 urllib.parse.urlencode()

 urllib.parse.urlparse("URL 문자열")
 urlparse() 함수
  아규먼트에 지정된 URL 문자열의 정보를 파싱하고 각각의 정보를 정해진
  속성으로 저장하여
  urllib.parse.ParseResult 객체를 리턴 함
  각 속성들을 이용하여 필요한 정보만 추출할 수 있음

 url1 = urlparse
 ('https://movie.daum.net/moviedb/main?movield=93252')
 RarseResult(scheme='https',
 netloc='movie.daum.net',
 path='/moviedb/main', params=",
 query='movield=93252',
 fragment=")

 url1.netloc, url1.path, url1.query, url1.scheme,
 url1.port, url1.fragment, url1.geturl()

 urllib.parse.urlencode()
 urlencode()함수
  메서드의 아규먼트로 지정된 name과 value로 구성된 딕셔너리 정보를
  정해진 규걱의 Query 문자열 또는 요청 파라미터 문자열로 리턴 함

 urlencode({'number':12524, 'type':'issue','action':'show'})
 number=12524&type=issue&action=show
 urlencode({'addr':'서울시 강남구 역삼동'})

 Query 문자열을 포함하여 요청
  Query 문자열을 포함하여 요청하는 것 => GET 방식 요청
  urllib.parse.urlencode 함수로
  name와 value로 구성되는 Query 문자열을 만듦
  URL 문자열의 뒤에 '?' 기호를 추가하여 요청 URL 사용
  parans = urllib.parse.urlencode({'name':'유니코','age':10})
  url = "http://unico2013.dothome.co.kr/"

 요청 파라미터를 포함하여 요청(1)
  요청 바디안에 요청 파라미터를 포함하여 요청하는 것
  => POST 방식 요청

 GET 방식과 같이 name과 value로 구성되는 문자열을 만듦
 POST 방식 요청에서는 바이트 형식의 문자열로 전달해야 하므로,
 encode('ascii') 메서드를 호출하여 바이트 형식의 문자열로 변경
 urllib.request.urlopen() 호출 시 바이트 형식의 문자열로 변경된
 데이터를 두 번째 아규먼트로 지정

 요청 파라미터를 포함하여 요청(2)
  URL 문자열과 요청 파라미터 문자열을 지정한 urllib.request.Request 객체 생성
  urllib.request.urlopen() 함수 호출 시 URL 문자열 대신 urllib.request.Request 객체 지정

