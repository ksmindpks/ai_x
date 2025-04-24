# functions.py = fn1~()부터 fn9~()
from customer import Customer

# 1.입력
def fn1_insert_customer_info():
    '''
    사용자로부터 name, phone, email, age, grade, etc를 입력받아 Customer형 객체 변환
    '''
    import re
    name = input('이름 : ')
    name_pattern = r'[가-힣]{2,}'
    while not re.search(name_pattern, name):
        print('이름을 제대로 입력하세요(한글 2글자 이상)')
        name = input('이름 : ')
    phone = input('전화번호 : ')
    email = input('이메일 : ')
    while True:
        try:
            age = int(input('나이 : '))
            if (age<0) | (age>130):
                raise Exception('나이 범위 이상')
            break
        except:
            print('올바른 나이를 입력하세요')

    try:
        grade = int(input('고객등급(1~5) : '))
        # grade = 1 if grade < 1 else 5 if grade> 5 else grade
        if grade < 1:
            grade = 1
        if grade > 5:
            grade = 5
    except:
        print('유효하지 않은 등급 입력시 1등급으로 초기화')
        grade = 1
        
    etc = input('기타정보 : ')

    return Customer(name, phone, email, age, grade, etc)

# 2.전체 출력
def fn2_print_customers(customer_list):
    'customer_list를 출력(pdf파일 40page 스타일)'
    print('='*70)
    print('{:^70}'.format('고객정보'))
    print('-'*70)
    print('{:>5}\t{:3}\t{:13}\t{:10}\t{:3}\t{}'.format('GRADE', '이름', '전화', '메일', '나이', '기타'))
    print('-'*70,) 
    
    for customer in customer_list:
        print(customer)
        
    print('='*70,'\n')

# 3.삭제 (동명이인이 있을 경우 해당 동명이인을 지울지 한사람 한사람 묻고 지우기 Y/N)
def fn3_delete_customer(customer_list):
    '''
    삭제하고자 하는 고객이름을 input받아
    매개변수로 들어온 custoemr_list에서 삭제하고 "삭제했음/삭제 못했음"을 메세지로 출력
    '''
    del_name = input('삭제할 이름을 입력하세요 :')
    del_idx = [] # 삭제할 인덱스를 저장하는 용도
    for idx, customer in enumerate(customer_list):
        if customer.name == del_name:
            del_idx.append(idx)

    if del_idx:
        for idx in del_idx[::-1]:
            print(customer_list[idx], '지우시겠습니까?', end='')
            answer = input('(Y/N)')
            if answer.upper() == 'Y': # (answwe = 'Y') | (answwe = 'y')와 같은 의미
                print('요청하신 {}({})님 삭제 완료'.format(del_name, 
                                                  customer_list[idx].phone))
                del customer_list[idx]
#     if del_idx:
#         for idx in del_idx[::-1]:
#             del customer_list[idx]
#         print('{}님 데이터 {}개를 삭제하였습니다'.format(del_name, len(del_idx)))
#     else:
#         print('{}님 데이터는 존재하지 않습니다'.format(del_name))

# 4.이름찾기 (동명이인이 있을 경우)
def fn4_search_customer(customer_list):
    '''
    찾고자 하는 이름을 input으로 받아, customer_list에서 검색하여
    같은 이름은 search_list에 append한 후 search_list를 출력, 
    같은 이름이 없으면 없다고 출력
    '''
    srch_name = input('출력할 이름을 입력하세요 :')
    srch_list = [] # 검색할 리스트를 저장하는 용도
    for customer in customer_list:
        if customer.name == srch_name:
            srch_list.append(customer)
            
    if srch_list:     
        print('{}님 데이터 {}개를 확인하였습니다'.format(srch_name, len(srch_list)))
        fn2_print_customers(srch_list)
    else:
        print('{}님 데이터는 존재하지 않습니다'.format(srch_name))

# 5.내보내기 (CSV)
def fn5_save_customer_csv(customer_list):
    '매개변수로 받은 customer_list를 딕셔너리리스트로 변환해서 csv 출력'
    import csv
    if customer_list:
        customer_dict = []
        for customer in customer_list:
            customer_dict.append(customer.as_dict())
            fieldnames_t = list(customer_dict[0].keys())
        filename = input('저장할 csv 파일명은?')
        with open('data/{}.csv'.format(filename), 'w', encoding='utf-8', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=fieldnames_t)
            dict_writer.writeheader() # header 쓰기
            dict_writer.writerows(customer_dict)

            print('data/{}.csv 내보내기 완료'.format(filename))
    else:
        print('입력된 회원 데이터가 없어 csv 내보내기 취소합니다')

# 9.종료 (종료하기 전 customer_list 를 txt 파일에 저장하고 종료)
def fn9_save_customer_txt(customer_list):
    '''customer_list를
    to_list_style()를 이용해서 ['홍길동, 010-8999-9999, e@e.com, 20, 3, 까칠해', ~]
    입력받으 파이명 txt로 백업
    '''
    customer_txt = []
    if customer_list:
        for customer in customer_list:
            customer_txt.append(customer.to_list_style() + '\n') # 스트링에 newline 추가
        filename = input('저장할 csv 파일명은(Default[Enter시]는 ch09_customers)?')
        if len(filename) == 0:
            filename = 'ch09_customers'
        with open('data/{}.txt'.format(filename), 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(customer_txt)
            
            print('data/{}.txt 내보내기 완료'.format(filename))
    else:
        print('입력된 회원 데이터가 없어 csv 내보내기 취소합니다')
