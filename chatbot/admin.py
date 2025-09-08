from django.contrib import admin
from .models import PdfDocument, Chat, Message, DocumentChunk

admin.site.register(PdfDocument)
admin.site.register(Chat)
admin.site.register(Message)
admin.site.register(DocumentChunk)

