import os
import json
import faiss
from django.conf import settings
from django.http import JsonResponse,HttpResponseBadRequest
from django.core.exceptions import PermissionDenied
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render,get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import PdfDocument,Chat,Message,DocumentChunk
from rag.main import DocLoader,Embedder,Retriever,AnswerGenerator
# Create your views here.

embedder=Embedder()


@login_required
def chat(request,user_id):
    if request.user.id!=user_id:
        raise PermissionDenied("You cant access this site..")
    return render(request,"chatbot.html",{'user_id':user_id})

@login_required
@csrf_exempt
@require_http_methods(['GET','POST'])
def chats_list_create(request):
    user=request.user
    if request.method=='GET':                                   #returns history
        chats=Chat.objects.filter(user=user).order_by('created_at')
        data=[{'chat_id':c.id,'title':c.title} for c in chats]
        return JsonResponse(data, safe=False)
    
    else:                                                       # preprocess uploaded pdf and create new chat
        if 'file' not in request.FILES:
            return HttpResponseBadRequest("Missing file")

        uploaded = request.FILES['file']
        title = request.POST.get('title') or uploaded.name

        doc = PdfDocument.objects.create(
            user=user,
            file=uploaded,
            upload_date=timezone.now()
        )

        #Preprocess Document
        doc_loader = DocLoader(doc.file.path)
        doc_loader.preprocess()
        chunks, page_num = doc_loader.split()

        # Save chunks to DB
        for text, page in zip(chunks, page_num):
            DocumentChunk.objects.create(document=doc, text=text, page_number=page)

        
        ret=Retriever(embedder)
        ret.build_index(chunks, page_num)

        index_dir=os.path.join(settings.MEDIA_ROOT,"indexes")
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, f"{doc.id}.index")
        faiss.write_index(ret.index, index_path)

        # Update PdfDocument
        doc.faiss_index.name = f"indexes/{doc.id}.index"
        doc.save()

        chat=Chat.objects.create(
            user=user,
            title=title,
            document=doc,
            created_at=timezone.now()
        )

        return JsonResponse({"id": chat.id, "title": chat.title, "created_at": chat.created_at,"doc_id":doc.id})

@csrf_exempt
@login_required 
def messages_list_create(request,chat_id):
    chat=get_object_or_404(Chat,id=chat_id,user=request.user)
    
    if request.method=="GET":                                       #returns list of messages in a chat
        messages=Message.objects.filter(chat_id=chat_id).order_by("timestamp")
        data=[
            {
                "id": m.id,
                "sender":m.sender,
                "content":m.content,

            }
            for m in messages
        ]
        return JsonResponse(data,safe=False)
    
    else:
        try:
            body = json.loads(request.body.decode("utf-8"))
        except json.JSONDecodeError:
            return HttpResponseBadRequest("Invalid JSON")
        print("Body Received: ",body)
        query = body.get("message")
        print("Query: ",query)
        if not query or not isinstance(query, str) or not query.strip():
            return HttpResponseBadRequest("Message must be a non-empty string")
        
        user_msg=Message.objects.create(
            chat_id=chat,
            sender="User",
            content=query,
            timestamp=timezone.now()
            )
        
        index_path = chat.document.faiss_index.path
        ret = Retriever(embedder)
        ret.index = faiss.read_index(index_path)

        doc_chunks = chat.document.chunks.all().order_by("id")
        ret.chunks = [c.text for c in doc_chunks]
        ret.page_num = [c.page_number for c in doc_chunks]

        retrieved_chunks,retrieved_pages=ret.retrieve(query)
        api_key = os.getenv("GEMINI_API_KEY")
        generator = AnswerGenerator(api_key=api_key)
        result=generator.generate(query, retrieved_chunks, retrieved_pages)
    

        bot_msg=Message.objects.create(
            chat_id=chat,
            sender="Bot",
            content=result,
            timestamp=timezone.now()
            )
        return JsonResponse({
            "user_message": {
                "id": user_msg.id,
                "sender": user_msg.sender,
                "content": user_msg.content,
                "timestamp": user_msg.timestamp,
            },
            "bot_message": {
                "id": bot_msg.id,
                "sender": bot_msg.sender,
                "content": bot_msg.content,
                "timestamp": bot_msg.timestamp,
            }
        })
