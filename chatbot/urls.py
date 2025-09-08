
from django.urls import path
from . import views
app_name='chatbot'

urlpatterns = [
    path('<int:user_id>/',views.chat,name='new_chat'),
    path('api/chats/', views.chats_list_create, name='chats_list_create'),
    path('api/chats/<int:chat_id>/messages/', views.messages_list_create, name='messages_list_create'),
    
]
