# customer.py = Customer클래스, to_customer(), load_customer()
class Customer:
    '고객데이터와 as_dict(), to_list_style (txt 백업시 필요), __str__()'
    def __init__(self, name, phone, email, age, grade, etc):
        self.name = name
        self.phone = phone
        self.email = email
        self.age = age
        self.grade = grade
        self.etc = etc
        pass
        
    def as_dict(self): # 객체를 딕셔너리데이터로 반환 (csv 파일 저장시)
        return {
            'name':self.name,
            'phone':self.phone,
            'email':self.email,
            'age':self.age,
            'grade':self.grade,
            'etc':self.etc
            }
    
    def to_list_style(self): #객체를 list return('홍길동, 010-8999-9999, e@e.com, 20, 3, 까칠해'), format이나 join함수 사용
        return '{}, {}, {}, {}, {}, {} '.format(
            self.name,
            self.phone,
            self.email,
            int(self.age),
            int(self.grade),
            self.etc)
#         temp = [self.name,
#             self.phone,
#             self.email,
#             str(self.age),
#             str(self.grade),
#             self.etc]
#         return ', '.join(temp)
         
    def __str__(self): #  return '***홍길동 010-8999-9999 e@e.com 20 까칠해'
        return '{:>5}\t{:3}\t{:13}\t{:10}\t{:3}\t{}'.format('*'*self.grade,
                                          self.name,
                                          self.phone,
                                          self.email,
                                          self.age,
                                          self.etc)

def to_customer(row):
    '''
        txt파일 내용 한줄 row ='홍길동 , 010-8999-9999, e@e.com, 20, 3, 까칠해' (txt파일 내용)을 
        매개변수로 받아 Customer 객체로 반환
    '''
    data = row.split(',')

    name = data[0]
    phone = data[1].strip()
    email = data[2].strip()
    age = int(data[3].strip()) # strip() : 좌우 공백(space, \t, \n) 제거
    grade = int(data[4].strip())
    etc = data[5].strip()
    return Customer(name, phone, email, age, grade, etc)

# 0.실행하면 data/ch09_customers.txt 파일의 내용을 load(customer_list)
#      data/ch09_customers.txt이 존재하지 않으면
#          빈 data/ch09_customers.txt 파일을 생성하고
#              데이터는 customer_list=[]

def load_customers():
    customer_list = []
    try:
        with open('data/ch09_customers.txt', 'r', encoding='utf-8') as txt_f:
            # txt 데이터 한줄씩 customer 객체로 받아 customer.append
            lines = txt_f.readlines()
            for line in lines:
                # line = '홍길동, 010-8999-9999, e@e.com, 30, 3, 까칠해'
                customer = to_customer(line)
                # print(customer)
                customer_list.append(customer)
            print('메모리에 upload 되었습니다.')

    except FileNotFoundError:
        with open('data/ch09_customers.txt', 'w', encoding='utf-8') as txt_f:
            print('초기화 파일을 생성했습니다')
    return customer_list