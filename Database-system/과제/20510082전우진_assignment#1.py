#!/usr/bin/env python
# coding: utf-8

# In[1]:


i=0
while i != 1000001: # i가 1,000,000까지만 돌아가도록 설정
    doc_1 = open("./temp.txt") # 0이 적힌 txt파일 오픈
    data = doc_1.readline() # 0 읽기
    temp_num = int(data) # 숫자 0으로 변환 후 저장
    doc_1.close() # 열었던 txt파일 닫음
    
    doc_2 = open("./temp.txt", "r+") # txt파일을 읽기 or 쓰기 모드로 열기
    add_num = temp_num + 1 # TEMP = TEMP + 1 연산
    add_num = str(add_num) # txt파일에 저장하기 위해 add_num을 문자열로 바꿔줌
    doc_2.write(add_num) # txt파일에 새로운 숫자로 저장
    doc_2.close() 
    i+=1
    
    # 진행률 체크
    if i%10000 == 0:
        print("{}/1000000".format(i))       

