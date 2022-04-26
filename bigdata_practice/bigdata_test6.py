from email import contentmanager
from matplotlib.pyplot import text


requests 패키지
 Kenneth Reitz에 의해 파이썬 라이브러리
 HTTP 프로토콜과 관련된 기능 지원

 requests 패키지
  아나콘다에는 requests 패키지가 site-packages로 설치되어 있음

 만일 설치를 해야 한다면 pipenv 명령으로 설치

 requests 패키지 및 urllib 패키지 차이
 urllib 패키지 / requests 패키지
 인코딩하여 바이너리 형태로 데이터 전송 / 딕셔너러 형태로 데이터 전송
 데이터 전달 방식에 따라 GET 요청, POST 요청을 구분 / 요청 메서드(GET, POST)를 명시하여 요청

 requests.request() 함수
  requests 패키지의 대표 함수
  HTTP 요청을 서버에 보내고 응답을 받아오는 기능 지원

  requests.request(method, url, **kwargs)

  method : 요청 방식 지정(GET, POST, HEAD, PUT, DELETE, OPTIONS)
  url : 요청할 대상 URL 문자열 지정
  parans : [선택적] 요청 시 전달할 Query 문자열 지정
  (딕셔너리, 투플리스트, 바이트열 가능)
  data : [선택적] 요청 시 바디에 담아서 전달할 요청 파라미터 지정
  json : [선택적] 요청 시 바디에 담아서 전달할 JSON 타입의 객체 지정
  auth : [선택적] 인증처리(로그인)에 사용할 튜플 지정

 HTTP 요청 방식을 지원하는 함수
  requests.request() 함수 외에 각각의 요청 방식에 따른 메서드들도 제공
  requests.request() 함수에 요청 방식을 지정하여 호출하는 것과 동일

 requests.get(url, params=None, **kwargs)
 requests.post(url, data=None, json=None, **kwargs)
 requests.head(url, **kwargs)
 requests.put(url, data=None, **kwargs)
 requests.patch(url, data=None, **kwargs)
 requests.delete(url, **kwargs)

 HTTP 요청 방식을 지원하는 함수
  HTTP 프로토콜에서 정의한 GET, POST, HEAD, PUT, PATCH, DELETE 등의
  요청 방식을 처리하는 메서드들을 모두 지원
  => GET, HEAD, POST만 학습

 GET 방식
  요청한 페이지의 헤더, 바디를 모두 받아오는 요청
  Query 문자열을 추가하여 요청할 수도 있음

 HEAD 방식
  콘텐츠 없이 요청 헤더만을 받아오는 방식
  요청 바디에 요청 파라미터 데이터를 추가하여 요청
  헤더와 바디를 모두 받아옴

 GET 방식 요청
  GET 방식 요청은 다음 두 가지 함수 중 하나를 호출하여 처리 가능
  requests.request('GET', url, **kwargs)
  requests.get(url, **kwargs)
  [kwargs]
  params - (선택적) 요청 시 전달할 Query 문자열을 지정
   Query 문자열을 포함하여 요청 : params 매개변수에 딕셔너리, 튜플리스트
   , 바이트열(bytes) 형식으로 전달
   Query 문자열을 포함하지 않는 요청 : params 매개변수의 설정 생략

 POST 방식 요청
  POST 방식 요청은 다음 두 가지 함수 중 하나를 호출하여 처리 가능
  request.request('POST', url, **kwargs)
  requests.post(url, **kwargs)
  [ kwargs ]
  data - (선택적) 요청 시 바디에 담아서 전달할 요청 +
  파라미터를 지정
  딕셔너리, 튜플리스트 형식, 바이트열(bytes) 형식
  json - (선택적) 요청 시 바디에 담아서 전달할 JSON 타입의 객체를 지정
  JSON 형식

 응답처리
  requests.request(), requests.get(), requests.head(),
  requests.post() 함수 모두 리턴 값은
  requests.models.Response 객체임

  text
   문자열 형식으로 응답 콘텐츠 추출
   추출 시 사용되는 문자 셋은 'ISO-8859-1'이므로 'utf-8'이나
   'euc-kr' 문자 셋으로 작성된 콘텐츠 추출 시 한글이 깨지는 현상 발생
   추출 전 응답되는 콘텐츠의 문자 셋 정보를 파악하여 Response 객체의
   encoding 속성에 문자 셋 정보를 설정한 후 추출

  content
   바이트열 형식으로 응답 콘텐츠 추출
   응답 콘텐츠가 이미지와 같은 바이너리 형식인 경우 사용
   한글이 들어간 문자열 형식인 경우 r.content.decode('utf-8')를 사용해서 디코드 해야 함

