from django.shortcuts import render,redirect
from .forms import SignupForm,LoginForm
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages


def home(request):
    return render(request,'home.html')


def login_user(request):
    if request.method == "POST":
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, "You have logged in successfully 🎉")
            return redirect("chatbot:new_chat",user_id=user.id)
        else:
            messages.error(request, "Invalid credentials ❌")
            return redirect("accounts:login")  
    else:
        form = LoginForm()
    return render(request, "login.html", {"form": form})


def signup_user(request):
    if request.method=="POST":
        form=SignupForm(request.POST)
        if form.is_valid():
            user=form.save()
            messages.success(request, "Account created successfully! Please log in.")
            login(request,user)
            return redirect('accounts:login')
    else:
        form = SignupForm()
    return render(request, 'signup.html', {'form': form})

def logout_user(request):
    logout(request)
    return redirect("accounts:home")

