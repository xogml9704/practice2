함수
 파이썬에서 기본적으로 제공
  print(), range(), type()
 모듈(module) 형식으로 제공
  예 : time.sleep()을 사용하기 위해 from errno import EHOSTUNREACH
from mailbox import NotEmptyError
from re import A
from hmac import digest
from inspect import formatargspec

from numpy import fromregex

from numpy import fromregex
import time
from winreg import HKEY_LOCAL_MACHINE
import time
import time
 사용자 정의
  나만의 함수

사용자 정의 함수
 기본형식
  def 함수명(입력 인자):
      명령문 1
      ( 명령문 2
      ...  )
      (return ...)
 입력값
  괄호 안에 내용이 있으면 입력값이 있다는 의미
 반환값
  함수에서 실행한 결과 등을 함수를 호출한 곳으로 돌려준다는 의미
   유형 입력값 출력값(반환값)
    1 X X
    2 X O
    3 O X
    4 O O

함수의 입력
 예 : double() 함수
  입력값으로 정수형값 3을 입력하면 double(num)의 num변수에 3이 저장
  함수는 그 값의 제곱을 계산하여 출력
  값을 화면에 출력만 할 뿐 함수 자체가 어떤 값을 반환하는 것은 아님
   def double(num):
       print(num, '의 제곱', num*num)
   double(3)
  결과
   3 의 제곱 : 9

단수 반환
 예 : add()
  입력값이 2개
  3은 변수 a에, 5는 변수 b에 각각 저장
  더한 결과가 sum에 저장
  return을 통해서 그 결과를 print() 함수 위치로 보냄

  def add(a, b):
      sum = a + b
      return sum
    print(add(3, 5))
  결과 : 8

복수 반환
 반환되는 값이 2개 이상일 때는 콤마(,)로 나열
 print() 함수에서는 2개의 값이 출력
 대부분의 다른 언어에서는 지원하지 않는 기능
 def add_sub(a, b):
     sum = a + b
     diff = a - b
     return sum, diff
 print(add_sub(3,5))
 
 결과 : (8, -2)

문자열
 따옴표
  문자열은 큰따옴표와 작은따옴표 모두 사용할 수 있음
  print("hello") # 'hello'를 출력!
  print('hello') # 'hello'를 출력!
  결과
  hello
  hello

 작은따옴표를 출력해야 하는 문자열은 큰따옴표를 사용
  print("My friend's house.")
  결과 : My friend' house.

 큰따옴표를 출력해야 하는 문자열은 작은따옴표를 사용
  print('그녀가 말했다. "안녕!.".')
  결과 : 그녀가 말했다. "안녕!".

문자열 읽어오기
 인덱스
  문자열을 변수에 저장하면 자동적으로 배열이 만들어짐
  animal = 'frog'
  print(animal[3])
  결과 : g

  n 개의 저장 장소가 있다면 인덱스는 0부터 n-1까지 존재

  파이썬에서는 마지막 항목에서부터 -1로 시작하여 1씩 감소하는 인덱싱 방법도 제공
  animal = 'frog'
  print(animal[-1])
  결과 : g

  한글에서도 같은 방식으로 동작

문자열 슬라이싱
 인자 형식
  [시작:끝+1:단계]
 인자가 1개인 경우
  animal = 'frog'
  print(animal[1])
  결과 : r
 인자가 2개인 경우
  animal = 'frog'
  print(animal[1:3])
  결과 : ro
 인자가 3개인 경우
  animal = 'frog'
  print(animal[0:3:2])
  결과 : fo

문자열 슬라이싱
 인자가 생략된 경우
  콜론(;)이 1개일 경우
  animal = 'frog'
  print(animal[:])
  print(animal[1:])
  print(animal[:2])
  결과
  frog # [:]는 전체의 문자열 출력
  rog  # [1:]는 인덱스 1에서부터 문자열의 끝까지 출력
  fr   # [:2]는 처음부터 2직전가지의 내용 출력

  콜론(:)이 2개일 경우
  animal = 'elephant'
  print(animal[::2])
  print(animal[::-2])
  결과
  eehn
  tapl

문자열의 병합
 2개의 문자열 사이에 '+' 기호를 넣으면 앞뒤의 문자열이 병합됨
 dog = '개'
 animal = '진돗' + dog
 print(animal)
 결과 : 진돗개

문자열의 함수(메소드)
 len() 함수
  괄호 안의 문자열(객체)의 길이를 반환
  animal = 'elephant'
  print(len(animal))
  결과 : 8

 메소드 호출
  문자열형 객체에서 제공하는 다양한 메소드(함수)를 점(.)으로 연결하여 호출
   객체, 메소드()
   animal = 'elephant'
   print('총 개수:', animal.count('e'))
   결과 : 총 개수 : 2

 정보수집
  animal = 'elephant'
  print('압쪽 찾기:', animal.find('e'))
  print('ep 찾기:', animal.find('ep'))
  print('뒤쪽 찾기:', animal.rfind('e'))
  print('위치:', animal.index('e'))
  print('el 시작:', animal.startswith('el'))
  앞쪽 e 찾기 : 0 # 문자나 문자열이 처음 나오는 인덱스 값을 반환
  ep 찾기 : 2
  뒤쪽 찾기 : 2 # 문자나 문자열이 가장 나중에 나오는 인덱스 값을 반환
  위치 : 0 # find()의 기능과 동일, 찾는 내용이 없으면 에러 발생
  el 시작 : True # 해당 문자열로 시작하면 True, 그렇지 않으면 False 출력

 in
  특정 문자 또는 문자열이 해당 문자열에 존재하는지 여부를 판단
   (문자 또는 문자열) (not) in 문자열

 정보수집
  문자열의 내용을 수정할 수 있는 메소드(함수)
  ai = 'python program'
  print('선택수정:', ai.replace('p', 'p'))
  print('소문자:', ai.lower())
  print('대문자:'. ai.upper())
  print('swap대소문자:'. ai.swapcase())
  print('첫문자만 대문자:', ai.capitalize())
  결과
  선택수정 : Python Program # 단어 교체
  소문자 : python program # 모든 문자들을 소문자로 변환
  대문자 : PYTHON PROGRAM # 모든 문자들을 대문자로 변환
  swap대소문자 : PYTHON PROGRAM # 모든 대/소문자를 역으로 변환
  첫문자만 대문자 : Python program # 문자열의 첫 문자만 대문자로 수정

 원래의 내용을 수정하려면 다음과 같이 명령문을 실행
  animal = 'Elephant'
  animal = animal.upper()
  print(animal)
  결과 : ELEPHANT

 정보 분할
  문자열의 공백을 삭제할 수 있는 메소드
   animal = ' elephant '
   print('왼쪽 벗겨내기:', animal.lstrip())
   print('오른쪽 벗겨내기:', animal.rstrip())
   print('좌우 벗겨내기:', animal.strip())
   결과
   왼쪽 벗겨내기 : elephant # 문자열의 왼쪽편 공란(들)을 삭제
   오른쪽 벗겨내기 :  elephant # 문자열의 오른쪽편 공락(들)을 삭제
   좌우 벗겨내기 : elephant # 문자열의 조우의 공란 (들)을 삭제

choice()
 무작위로 문자를 출력
 import random
 chars = ['한', '글', '우', '수']
 print(random.choice(chars))
 결과 : 수

shuffle()
 실제로 배열의 순서가 바뀌어 저장됨
 import random
 chars = ['한', '글', '우', '수']
 random.shuffle(chars)
 print(chars)
 ['우', '수', '한', '글'] # 4개의 글자가 무작위의 순으로 재벼열 됨

배열형 자료구조
 배열형 자료구조
  문자열형, 유니코드 문자열형, 리스트, 튜플, 바이트배열, xrange()
  한 개의 변수로 다수 개의 데이터를 저장해 두고 편리하게 접근

mutable과 immutable
 변경할 수 있는 데이터형(mutable)
  리스트형(list)
  사전형(dict)
  집합형(set)
 변경할 수 없는 데이터형(immutable)
  숫자형 : 정수형(int), 실수형(float)
  부울형(bool)
  문자열형(str)
  튜플형(tuple)
 정수형은 immutable

리스트
 다수의 자료형들도 입력 가능
 대괄호([])를 사용하며 각각의 항목들은 콤마(,)로 구분

 1차원 리스트
  price = [1020, 870, 3160, 2650]
  fruits = ['사과', '오렌지', '포도', '복숭아']
  print(price)
  print(fruits)
  결과
  [1020, 870, 3160, 2650]
  ['사과', '오렌지', '포도', '복숭아']

  price = [1020, 870, 3160, 2650]
  fruits = ['사과', '오렌지', '포도', '복숭아']
  print(price[1])
  print(fruits[-1])
  결과
  870
  복숭아

 리스트의 복사
  a를 b에 복사하고 b의 index 0번에 저장된 값을 변경하면
  a = [3, 5, 7]
  b = a
  b[0] = b[0] - 2
  print('a=', a, 'b=', b)

  a의 배열까지 수정됨
  결과
  a = [1, 5 ,7] b = [1, 5, 7]

  같은 객체를 두 개의 다른 변수들이 가리킴

리스트에서의 병합 및 삽입
 복수 개 변수의 병합
 x = 12.23
 y = 23.34
 packing = [x, y] # packing!
 type(packing)
 print('Packing:', packing)
 [c1, c2] = packing # packing!
 print('Unpacking:\ncl:', c1)
 print('c2:', c2)
 결과
 Packing: [12.23, 23.34]
 Unpacking:
 c1: 12.23
 c2: 23.34

 복수 개 리스트의 병합
 fruits1 = ['사과', '오렌지', '포도']
 fruits2 = ['복숭아', '키위']
 allfruits = fruits1 + fruits2
 print(allfruits)
 결과
 ['사과', '오렌지', '포도', '복숭아']

리스트에 원소 삽입
 리스트의 함축(comprehension)
  특정 리스트에 저장된 모든 원소들에 대해 조건에 맞는 원소들만을 선택적으로 추가
  리스트(1)의 원소들을 i(2)로 읽어 와서 조건식(3)에서 그 값을 테스트한 후 결과가 참이면
  i(4)를 리스트에 입력
  [i for i in (리스트명) if (조건식)]
   4     2        1            3
 mylist = [3, 5, 4, 9, 1, 8, 2, 1]
 newlist = [i for i in mylist if (1%2)==0]
 print(newlist)
 결과
 [4, 8, 2]

존재 여부
 in
  문자열에 특정한 문자가 있는지 검사
  word = 'orange'
  print('r' in word)
  결과
  True
 
 not in
  예 : 리스트 fruits에 특정한 과일이 없는지 검사
  fruits = ['사과', '오렌지', '포도']
  print('포도' not in fruits)
  결과
  False
튜플
 초기화한 후 변경할 수 없는 배열
 괄호(())로 묶인 형식으로 항목들은 콤마(,)로 연결
 괄호를 생략해도 됨
 문자열, 숫자, 튜플, 리스트를 포함할 수 있음
 인덱싱, 슬라이싱, 메소드 기능이 리스트와 유사
 empty = () # 빈 튜플을 생성 또는 empty = tuple()
 animals = ('토끼', '사자', '오렌지')
 fruits = '사과', '오렌지', 1020, 880
 start = '하나', '둘'
 print(fruits)
 결과
 ('사과', '오렌지', 1020, 880)

 인덱싱, 슬라이싱
 print(fruits[1]) # 인덱싱 기능
 print(fruits[1:3]) # 슬라이싱 기능
 결과
 오렌지
 ('오렌지', 1020)

 한번 입력한 내용을 변경할 수 없음
 fruits[1] = '키위'
 결과 : 에러 발생

 중첩
 animals = ('토끼', '사자', '원숭이')
 fruits = '사과', '오렌지', 1020, 88
 things = animals, fruits
 print(things)
 (('토끼', '사자', '원숭이'), ('사과', '오렌지', 1020, 88))

 리스트를 포함하는 튜플
 fruits = (['포도', '망고'], ['사과', '키위'])
 print(fruits[1])
 ['사과', '키위']
  튜플에는 새로운 값을 입력할 수 없음
  fruits = (['포도', '망고'], ['사과', '키위'])
  newfruits = ['수박', '참외']
  fruits[1] = newfruits
  결과 : 에러발생!

  튜플 내부의 원소로 존재하는 리스트는 수정 가능
  fruits = (['포도', '망고'], ['사과', '키위'])
  fruits[1][0] = '수박'
  print(fruits)
  결과
  (['포도', '망고'], ['수박', '키위'])

 튜플을 포함하는 리스트
  리스트의 원소를 수정하는 것은 가능
  (튜플 내용을 수정한 것이 아님)
  fruits = [('포도', '망고'), ('사과', '키위')]
  fruits[0] = ('수박', '참외')
  print(fruits)
  결과
  [('수박', '참외'), ('사과', '키위')]

사전
 키(key)와 값(value)의 쌍(pair)
 키로 색인(문자열, 숫자, 튜플)
 하나의 사전에 유일한 키들을 포함하고 있어야 함
 직,간접적으로 수정이 가능한 객체를 포함하고 있다면 키로 사용할 수 없음

사전의 생성
 중괄호 {}는 빈 사전을 생성

삽입
 정렬되지 않은 키:값 {key:value} 형태의 쌍으로 구성된 집합 형식으로 입력
 인덱스에 순서를 나타내는 번호를 사용하지 않고 키를 인덱스 처럼 사용
 사전에 2개 이상의 항목이 있으면 콤마(,)로 연결됨

 student = {} # 빈 사전을 생성함. 또는 student = dict()
 student['지훈'] = 1234
 print(student)
 결과
 {'지훈': 1234}

 student['수민'] = 2345
 print(student)
 {'지훈' : 1234, '수민': 2345}

어러 개의 정보를 하나의 명령문으로 입력
 fruitdb = {'사과':1020, '오렌지':880, '포도':3160}
 print(fruitdb)
 결과
 {'사과':1020, '오렌지':880, '포도':3160}

인덱싱과 슬라이싱
 인덱싱이나 슬라이싱 모두 동작하지 않고 에러를 발생시킴
 fruitdb = {'사과':1020, '오렌지':880, '포도':3160}
 print(fruitdb[1])
 print(fruitdb[1:2])
 결과
 # 에러 발생!
 # 에러 발생!

사전 항목 삭제
 del 명령
 fruitdb = {'사과':1020, '오렌지':880, '포도':3160}
 del fruitdb['사과']
 print(fruitdb)
 결과
 {'오렌지':880, '포도':3160}

사전항목검색
 키 in 사전 : 키값이 있으면 True 없으면 False
 get() : 키가 있으면 해당 키의 값을 반환하며 없으면 None
 keys() : 모든 키를 변환
 values() : 모든 값을 반환
 items() : 모든 키 : 값 쌍들을 반환
 student = {'현준' : 1234, '민지' : 2345}
 print('SeJong' in student)
 print(student.get('현준'))
 print(student.get('민지'))
 print(student.keys())
 print(student.values())
 print(student.items())
 결과
 False # (키 in 사전) 
 1234 # 키로 값을 출력
 2345 # 키로 값을 출력
 dict_keys(['현준', '민지']) # 키틀만 출력
 dict_values([1234, 2345]) # 값들만 출력
 dict_items([('현준', 1234), ('민지', 2345)]) # items() 메소드

사전 병합
 update() 메소드를 이용
 student = {'현준' : 1234, '민지' : 2345, '승민' : 3456, '유진' : 4567}
 
집합(Set)
 중복되지 않고 정렬되지 않은 원소들로 구성됨
 사전과 같이 {} 기호를 사용하고 원소들을 콤마(,)로 구분
 사전과 달리 키만 있고 값이 없는 형식
 dict = { } # 빈 사전을 생성 또는 dict = set()
 dict = {3, 2, 3, 1} # 중복된 원소를 포함한 데이터 입력
 print(dict)
 결과 {1, 2, 3}

 set() 함수를 이용하여 리스트나 튜플을 집합 형태로 생성 할 수 있음
 fruits = ['사과', '오렌지', '포도', '오렌지']
 fruits = set(fruits)
 print(fruits)
 결과
 ['오렌지', '사과', '포도']

집합의 원소 추가 및 삭제
 추가
  add()
  fruits = {'사과', '오렌지', '포도'}
  fruits.add('키위')
  print(fruits)
  결과
  {'키위', '사과', '포도', '오렌지'}
  update()
  fruits = {'사과', '오렌지', '포도'}
  fruits.update({'수박', '배'})
  print(fruits)
  결과
  {'사과', '포도', '오렌지', '수박', '배'}

 삭제
  remove()
  pop()
  fruits = {'사과', '오렌지', '포도', '수박'}
  fruits.remove('오렌지')
  print(fruits)
  fruits.pop()
  print(fruits)
  fruits.clear()
  print(fruits)
  결과
  {'사과', '수박', '포도'}
  {'수박', '포도'}
  set()           # 집합에 원소가 비어있다는 의미

집합
 존재여부
  해당하는 값이 멤버인지 확인하려면 'in'이나 'not in' 사용
  fruits = {'사과', '오렌지', '포도'}
  print('사과' in fruits)
  print('키위' not in fruits)
  결과
  True
  True

연산자 / 기능 / 설명
| / 합집합(union) / 두 집합의 모든 원소
& / 교집합(intersection) / 두 집합에서 공통적으로 가지고 있는 원소
- / 차집합(difference) / 왼쪽 집합에서 오른쪽 집합의 원소를 뺀 것
^ / 배타적 차집합(symmetric difference) / 공통된 원소를 제외한 모든 원소

one = {1, 3, 5, 7, 8}
two = {1, 3, 5, 6, 8}
print('one | two:', one | two)
print('one & two:', one & two)
print('one - two:', one - two)
print('one ^ two:', one ^ two)
결과
one | two : {1, 3, 5, 6, 7, 8}
one & two : {1, 3, 5, 6, 7, 8}
one - two : {7}
one ^ two : {6, 7}

집합의 관계 연산자
 연산자 / 기능 / 설명
 <= / 부분집합(subset) 부분집합 / 왼쪽집합의 모든 원소가 오른쪽에 있는지 조사
 < / 부분집합(subset) 진부분집합 / 왼쪽집합의 모든 원소가 오른쪽에 있는지 조사(단, 2개는 같지 않아야 한다.)
 >= / 상위집합(superset) 상위집합 / 오른쪽집합의 모든 원소가 왼쪽에 있는지 조사
 > / 상위집합(superset) 진상위집합 / 오른쪽집합의 모든 원소가 왼쪽에 있는지 조사(단, 2개는 같지 않아야 한다.)

 one = {1, 3, 5 ,8}
 two = {1, 3, 5, 8}
 print('one <= two : ', one <= two)
 print('one < two :', one < two)
 print('one >= two :', one >= two)
 print('one > two : ', one > two)
 결과
 one <= two : True
 one < two : False
 one >= two : True
 one > two : False

