from django.shortcuts import render, get_object_or_404, redirect
from .models import Student
from django.contrib import messages

# Create your views here.
def list(request):
    students = Student.objects.all()
    return render(request, "student/list.html", {"students": students})

def get(request, id):
    # student = Student.objects.get(id=id)
    # return render(request, "student/get.html", {"student": student})
    # student = get_object_or_404(Student, id=id) # 해당 id없을 경우 404에러
    # return render(request, "student/get.html", {"student": student})
    try:
        student = Student.objects.get(id=id)
        return render(request, "student/get.html", {"student": student})
    except Student.DoesNotExist:
        # 존재하지 않는 id로 검색할 경우
        messages.error(request, f"{id}번 학생을 찾을 수 없습니다.")
        # return redirect("/student")
        return redirect("student:list")
    
def delete(request, id: int):
    student = Student.objects.filter(id=id) # 없는 id일 경우 빈 list
    if student:
        student.delete()
        messages.success(request, f"{id}번 학생을 삭제하였습니다")
        return redirect("student:list")
    else:
        # 존재하지 않는 id로 검색할 경우
        messages.error(request, f"{id}번 학생은 존재하지 않습니다")
        return redirect("student:list")