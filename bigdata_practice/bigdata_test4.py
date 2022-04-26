from msilib import sequence
from xml.etree.ElementTree import QName


DNA 분석 과정
 DNA 서열을 아미노산의 서열로 변환
 12개의 알파벳으로 구성되어 있는 DNA 데이터를 3개씩 잘라서 하나의 아미노산으로 변환
 DNA ATA CAA TGG CAA
 아미노산 I Q W Q

데이터 다운로드
 DNA와 프로틴 시퀀스 데이터를 다운로드
  NCBI 접속
  NM_207618.2를 검색 (search)
  Nucleotide sequence(뉴클레오티드 서열) 다운로드
  "NM_207618.2.fna"

파이썬으로 DNA 데이터 가공
 주피터 노트북 사용
 open() 명령을 이용하여 읽기 전용으로 파일 열기
 내용을 출력
 f = open("NM_207618.2.fna", "r")
 sequence = f.read()
 sequence

 불 필요한 행 삭제
 with open("NM_207618.2.fna","r") as inf:
     data = int.read().splitlines(True)
 with open("dna1.txt", "w") as outf:
     outf.writelines(data[1:])
 f=open("dna1.txt", "r")
 sequence = f.read()
 sequence

 불 필요한 문자 삭제
 '\ n', '\ r', ' '등
 sequence = sequence.replace('\n', '')
 sequence = sequence.replace('\r', '')
 sequence = sequence.replace(' ', '')

 