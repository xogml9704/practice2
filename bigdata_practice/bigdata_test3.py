from fractions import Fraction
from matplotlib.cbook import report_memory
from sklearn.metrics import balanced_accuracy_score


클래스
 객체지향의 가장 기본적 개념
 관련된 속성과 동작을 하나의 범주로 묶어 실세계의 사물을 흉내냄

 모델링
  사물 분석하여 필요한 속성과 동작 추출

 캡슐화
  모델링 결과를 클래스로 포장
  
 balance = 8000
 def deposit(money):
     global balance
     balance += money

 def inquire():
     print("잔액은 %d원입니다.", %balance)

 deposit(1000)
 inquire()

 결과
 잔액은 9000원 입니다.

 사물의 속성은 변수로, 동작은 함수로 표현

 멤버
  클래스 구성하는 변수 (& 메소드)

 메서드
  클래스에 소속된 함수

 생성자
  클래스 선언 형식
  class 이름 :
      def __init__(self, 초기값):
          멤버 초기화
      매서드 정의

  __init__ 생성자
   통상 객체 초기화
   class Human:
       def __init__(self, age, name):
           Self.age = age
           Self.name = name
       def intro(self):
           print(str(Self.age) + "살 " + self.name + "입니다. ")
   kim = Human(29, "김상형")
   kim.intro()
   lee = Human(45, "이승우")
   lee.intro()
   결과
   29살 김상형입니다.
   45살 이승우입니다.

 객체 생성 구문
  객체를 __init__의 첫 번째 인수로 self로 전달
  객체 = 클래스명(인수)
  생성문에서 전달한 인수를 두 번째 인수로 전달
  새로 생성되는 객체 멤버에 대입

  메서드는 필요한 만큼 선언할 수 있음
   객체, 메서드()

 상속
  기존 클래스 확장하여 멤버 추가하거나 동작 변경
   클래스 이름 다음의 괄호 안에 부모 클래스 이름 지정
    class 이름(부모):
        ...

   class Human:
       def __init__(Self, age, name):
           Self.age = age
           Self.name = name

       def intro(Self):
           print(str(Self.age) + "살 " + Self.name + "입니다")
    
   class Student(Human):
       def __init__(Self, age, name, stunum):
           super().__init__(age, name)
           Self.stunum = stunum
    
       def intro(Self):
           super().intro()
           print("학번 : "+ str(Self.stunum))
       def study(Self):
           print("하늘천 따지 검을현 누를황")

   kim = Human(29, "김상형")
   kim.intro()
   lee = Student(34, "이승우", 930011)
   lee.intro()
   lee.study()
   결과
   29살 김상형입니다.
   34살 이승우입니다.
   학번 : 930011
   하늘천 따지 검을현 누를황

 super() 메서드
  자식 클래스에서 부모의 메서드 호출할 때 사용

 엑세서
  파이썬 클래스의 멤버는 모두 공개되어 누구나 외부에서 액세스 가능
  일정한 규칙 마련하고 안전한 액세스 보장
   게터(Getter) 메서드
    - 멤버 값 대신 읽음
   세터(Setter) 메서드
    - 멤버 값 변경

  class Date:
      def __init__(Self, month):
          Self.month = month
      def getMonth(Self):
          return Self.month
      def SetMonth(Self, month):
          if 1 <= month <= 12:
              Self.month = month

  today = Date(8)
  today.Setmonth(15)
  print(today.getmonth())
  결과 8

 클래스 메서드
  특정 객체에 대한 작업 처리하는 것이 아니라 클래스 전체에 공유
  @classmethod
  첫 번쨰 인수로 클래스에 해당하는 cls 인수
  class Car:
      count = 0
      def __init__(Self, name):
          Self.name = name
          Car.count += 1
      @classmethod
      def outcount(cls):
          print(cls.count)

  pride = Car("프라이드")
  korando = Car("코란도")
  Car.outcount()
  결과 : 2

 정적 메서드
  클래스에 포함되는 단순 유틸리티 메서드
  특정 객체에 소속되거나 콜래스 관련 동작 하지 않음
  @staticmethod

  class Car:
      @staticmethod
      def hello():
          print("오늘도 안전 운행 합시다.")
      count = 0
      def __init__(Self, name):
          Self.name = name
          Car.count += 1
      @classmethod
      def outcount(cls):
          print(cls.count)

  Car.hello()
  결과 : 오늘도 안전 운행 합시다.

연산자 메서드
 연산자 사용하여 객체끼리 연산
 연산자 오버로딩
  클래스별로 연산자 동작을 고유하게 정의

 연산자 / 메서드 / 우변일 때의 메서드
 == / __eq__
 != / __ne__
 < / __it__
 > / __gt__
 <= / __le__
 >= / __ge__
 + / __add__ / __radd__
 - / __sub__ / __rsub__
 * / __mul__ / __rmul__
 / / __div__ / __rdiv__
 /(division 임포트) / __truediv__ / __rtruediv__
 // / __floordiv__ / __rfloordiv__
 % / __mod__ / __rmod__
 ** / __pow__ / __rpow__
 << / __Ishlft__ / __rlshift__
 >> / __rshift__ / __lshift__

 class Human:
     def __init__(Self, age, name):
         Self.age = age
         Self.name = name
     def __eq__(Self, other):
         return Self.age == other.age and Self.name == other.name
 
 kim = Human(29, "김상형")
 sang = Human(29, "김상형")
 moon = Human(44, "문종민")
 print(kim == sang)
 print(kim == moon)
 결과
 True
 False

 특수 메서드
  특정한 구문에 객체 사용될 경우 미리 약속된 작업 수행
  메서드 / 설명
  __str__ / str(객체) 형식으로 객체를 문자열화한다.
  __repr__ / repr(객체) 형식으로 객체의 표현식을 만든다.
  __len__ / len(객체) 형식으로 객체의 길이를 조사한다.

  class Human:
      def __init__(Self, age, name):
          Self.age = age
          Self.name = name
      def __str__(Self):
          return "이름 %s, 나이 %d" % (Self.name, Self.age)

  kim = Human(29, "김상형")
  print(kim)
  결과
  이름 김상형, 나이 29

유틸리티 클래스
 Decimal
  정수 혹은 문자열, 실수로 초기화
  오차 없이 정확하게 10진 실수를 표현
   컴퓨터에서 이진 실수로 십진 실수를 정확하게 표현하기 어려움
  f = 0.1
  sum = 0
  for i in range(100):
      sum += f
  print(sum)
  결과
  9.99999999999999

  from decimal import Decimal

  f = Decimal('0.1')
  sum = 0
  for i in range(100):
      sum += f
  print(sum)
  결과
  10.0

 Context 객체
  연산 수행 방법을 지정
  getcontext / setcontext 함수로 컨텍스트 변경
  같은 연산이라도 컨텍스트 따라 결과 다를 수 있음

 컨텍스트 / 설명
 BasicContext / 유효 자리수 9, ROUND_HALF_UP 반올림
 ExtendedContext / 유효 자리수 9, ROUND_HALF_EVEN 반올림 처리
 DefaultContext / 유효 자리수 28, ROUND_HALF_EVEN 반올림 처리

 Fraction
  유리수를 표현
   분모와 분자를 따로 전달하여 분수 형태 숫자 표현함
   Fraction([부호] 분자, 분모)

  from fractions import *

  a = Fraction(1, 3)
  print(a)
  b = Fraction(8, 14)
  print(b)
  결과
  1/3
  4/7

 array 모듈
  동일 타입 집합인 배열을 지원
  대량 자료를 메모리 낭비 없이 저장 및 고속 액세스 가능
  array(타입코드, [초기값])
  타입 / C 타입 / 설명
  b, B / char / 1바이트의 정수
  u /  / 2바이트의 유니코드 문자(3,3 이후 지원 안 함)
  h, H, i, l / shirt, int / 2바이트의 정수
  I, L / long / 4바이트의 정수
  q, Q / long long, __int64 / 8바이트의 정수(3.3이상에서만 지원)
  f / float / 4바이트의 실수
  d / double / 8바이트의 실수

 모듈
  모듈의 작성 및 사용
  모듈
   파이썬 코드를 저장하는 기본 단위
   편의상 스크립트를 여러 개의 파일로 나눈 하나
   .py 빼고 파일명으로 불림
   파이썬에서 자주 사용하는 기능은 표준 모듈로 제공됨
   직접 제작 가능
   INCH = 2.54

   def calcsum(n):
       sum = 0
       for num in range(n + 1):
           sum += num
       return sum

  import util

  print("linch =", util.INCH)
  print("~10 =", util.calcsum(10))
  결과
  linch = 2.54
  ~10 = 55

 테스트 코드
  모듈에 간단한 테스트 코드를 작성할 수 있음
  INCH = 2.54

  def calcsum(n):
      sum = 0
      for num in range(n + 1):
          sum += sum
      return sum

  print("인치 =", INCH)
  print("합계 =", calcsum(10))
  결과
  인치 = 2.54
  합계 = 55

 타 모듈에 함수 제공하는 모듈에서는 테스트 코드를 조건문으로 감쌈
 INCH = 2.54
 def calcsum(n):
     sum = 0
     for num in range(n + 1):
         sum += num
     return sum

 if __name__=="__main__":
     print("인치 =", INCH)
     print("합계 =", calcsum(10))

 import util2
 print("linch =", util2.INCH)
 print("~10 =", util2.calcsum(10))

 모듈 경로
  모듈은 임포트하는 파일과 같은 디렉토리에 있어야 함

  모듈은 특정 폴더에 두려면 임포트 패스에 추가

패키지
 모듈을 담는 디렉토리
 디렉토리로 계층 구성하면 모듈을 기능 등에 따라 분류 가능

 import sys
 sys.path.append("C:/PyStudy")

 import mypack.calc.add
 mypack.calc.add.outadd(1,2)

 import mypack.report.table
 mypack.report.table.outreport()

 결과
 3
 -------
 report
 -------

 함수명이나 모듈명에 충돌 발생하지 않음
 단일 모듈에 비해 호출문 길어지는 불편함 있음
 -from 패키지 import 모듈
 from mypack.calc import add
 add.outadd(1,2)

 __init__.py
 모든 모듈 한꺼번에 불러올 때는 어떤 모듈 대상인지 밝혀두어야 임포트할
 대상 모듈 리스트 명시

 패키지 로드될 떄의 초기화 코드 작성해 둠
 import sys
 sys.path.append("C:/PyStudy")

 from mypack.calc import *
 add.outadd(1,2)
 multi.outmulti(1,2)

 __all__ = ["add", "multi"]
 print("add module imported")

 import * 로 읽을 떄 add 및 multi 모듈 모두 읽어 옴
 __init__.py의 초기화 코드 실행

 __init__.py 목록에 어떤 모듈 작성할 것인지는 패키지 개발자 재량

 한 번 임포트한 모듈을 컴파일 상태로 캐시에 저장
  확장자 pyc
  한 모듈을 각각 다른 파이썬 버전끼리 공유 가능

서드 파티 모듈
 모듈의 내부
  각 모듈은 기능별로 나누여져 다수 함수 포함

  dir 내장 함수
   모듈에 있는 함수나 변수 목록 조사

 외부 모듈의 목록
  서드 파티 모듈(Third Party Module)
   파이썬 외 회사 및 단체가 제작하여 배포하는 모듈
   모듈 / 설명
   Django, Flask / 웹 프레임워크
   BeaultulSoup / HTML, XML 피서
   wxPytho, PyGik / 그래픽 툴킷
   pyGame / 게임 제작 프레임워크
   PIL / 이미지 처리 라이브러리
   pyLibrary / 유틸리티 라이브러리

 pip (Python Package Index)
  외부 모듈 관리 용이
  pip 명령 패키지명
  명령 / 설명
  install / 패키지를 설치한다
  uninstall / 설치한 패키지를 삭제한다.
  freeze / 설치한 패키지 목록을 보여 준다.
  show / 패키지의 정보를 보여 준다.
  search / pyPI에서 패키지를 검색한다.

 import wx
 app = wx.App()
 frame = wx.Frane(None, 0, "파이썬 만세")

 frame.Show(True)
 app.MainLoop()