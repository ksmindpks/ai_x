#
from customer import load_customers
from functions import fn1_insert_customer_info, fn2_print_customers
from functions import fn3_delete_customer, fn4_search_customer, fn5_save_customer_csv, fn9_save_customer_txt

def main():
    # global customer_list
    customer_list = load_customers() # ch09_customers.txt의 내용을 load
    while True:
        print("1:입력 ","2:전체출력 ","3:삭제 ","4:이름찾기 ","5:내보내기 (CSV)", "9:종료 ", sep=' | ', end='')
        fn = input('메뉴선택):')
        if fn == '1':
            customer = fn1_insert_customer_info() # 입력받은 내용으로 customer 객체를 반환
            customer_list.append(customer)
            print('fn1_호출한 결과를 customer에 받아 customer_list에 append')
        elif fn == '2':
            fn2_print_customers(customer_list)
            print('fn2_호출한 결과, 전체 출력')
        elif fn == '3':
            fn3_delete_customer(customer_list)
            print('fn3_호출한 결과, 항목 삭제')
        elif fn == '4':
            fn4_search_customer(customer_list)
            print('fn4_호출한 결과, 검색 항목 출력')
        elif fn == '5':
            fn5_save_customer_csv(customer_list)
            print('fn5_호출한 결과, 지정한 이름으로 csv 파일 생성')
        elif fn == '9':
            fn9_save_customer_txt(customer_list)
            print('fn2_호출한 결과, 백업파일(.txt) 저장 후 종료')
            break
        else:
            print('올바른 메뉴 번호를 선택 바랍니다.')

# main() 함수 호출
if __name__ == '__main__':
    main()