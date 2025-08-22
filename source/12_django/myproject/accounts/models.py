from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Profile(models.Model):
    user = models.OneToOneField(User,
                                on_delete=models.CASCADE) # User가 삭제될 때, profile은 어떻게 할
    phone_number = models.CharField(verbose_name="전화", max_length=20)
    address = models.CharField(verbose_name="주소", max_length=200)
    def __str__(self):
        return "{}({}-{})".format(self.user.username,
                                  self.phone_number,
                                  self.address)

# 이벤트저리 == signals사용(post_save) : profile.save()성공시 가입인사를 메일 전송
from django.db.models.signals import post_save
from django.core.mail import send_mail

def on_send_mail(sender, **kwargs):
    print('♣ on_send_mail', kwargs)
    if kwargs['created']: # True면 새로 생성된 것, False면 수정된 것
        user = kwargs['instance'].user
        if not user.email: # 회원가입시 메일 입력 안함
            print("이메일 주소가 없어서 메일 못 보내요.")
            return
        # 메일 보내기
        subject = f"{user.username} 회원가입해 주셔서 감사합니다.(메일제목)"
        body = f"{user.username}님, 회원가입을 축하합니다.\n\n" 
        bodyHtml = f"<h1>{user.username}님, 회원가입을 축하합니다.</h1>"
        # settings.py에 반드시
        send_mail(
            subject=subject, 
            message=body, 
            from_email='ksparkp@gmail.com',
            recipient_list=[user.email],
            html_message=bodyHtml,
            fail_silently=False, # 메일 전송이 안 되었을 경우, 아무일도 하지 않음
        )
# on_send_mail 함수는 Profile 모델이 저장될 때마다 호출됨
post_save.connect(on_send_mail, sender=Profile)




