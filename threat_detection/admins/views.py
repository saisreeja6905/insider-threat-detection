from django.shortcuts import render,redirect
from django.contrib import messages
from users.models import User_Registration
# Create your views here.

def adminlogin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username == 'admin' and password == 'admin':
            return render(request, 'admins/adminhome.html')
        else:
            messages.error(request, 'Invalid username or password.')
            return redirect('adminlogin')
    return render(request, 'adminlogin.html')

def adminhome(request):
    return render(request, 'admins/adminhome.html')

def view_users(request):
    data=User_Registration.objects.all()
    return render(request, 'admins/view_users.html',{'data':data})


def activate_user(request, user_id):
    user = User_Registration.objects.get(id=user_id)
    user.status = 'active'
    user.save()
    return redirect('view_users')

def deactivate_user(request, user_id):
    user = User_Registration.objects.get(id=user_id)
    user.status = 'waiting'
    user.save()
    return redirect('view_users')