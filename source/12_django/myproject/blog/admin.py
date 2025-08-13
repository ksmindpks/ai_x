from django.contrib import admin
from blog.models import Post, Comment, Tag
# Register your models here.
admin.site.register(Post) # admin 페이지에서 POST 테이블 엑세스 가능
admin.site.register(Comment) # admin 페이지에서 Comments
admin.site.register(Tag)
