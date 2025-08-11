from django.urls import path, register_converter
from . import views
from .converters import IdConverter
# "/student" -> student list 출력 (student:list)
# "/student/get/2" -> id=2인 student 상세보기 (student:get)
# "/student/del/2" -> id=2인 student 삭세 (student:del)
app_name = "student"
register_converter(IdConverter, "dddd")
urlpatterns = [
    path("", views.list, name="list"),
    path("get/<dddd:id>/", views.get, name="get"),
    path("delete/<int:id>/", views.delete, name="delete")
]
