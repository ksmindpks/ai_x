from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def home(request):
    # return HttpResponse("<h1>Hello, Django~~~~~~~~~~~~~~</h1>")
    return render(request=request,
                  template_name="home.html",
                  context={"msg": "Django(장고)"})