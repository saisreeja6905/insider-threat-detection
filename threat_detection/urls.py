"""
URL configuration for threat_detection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from users import views as user_views
from admins import views as admin_views
from eda import graphs

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', user_views.home, name='home'),

    path('adminlogin/', admin_views.adminlogin, name='adminlogin'),
    path('adminhome/', admin_views.adminhome, name='adminhome'),
    path('view_users/', admin_views.view_users, name='view_users'),
    path('activate_user/<int:user_id>/', admin_views.activate_user, name='activate_user'),
    path('deactivate_user/<int:user_id>/', admin_views.deactivate_user, name='deactivate_user'),

    path('register/', user_views.register, name='register'),
    path('userlogin/', user_views.userlogin, name='userlogin'),
    path('userhome/', user_views.userhome, name='userhome'),
    path('training', user_views.train_models, name='training'),
    path('predict', user_views.predict_attack, name='predict'),

    path('graphs/', graphs, name='graphs'),
]
