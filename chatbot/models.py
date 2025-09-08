from django.db import models
from accounts.models import User

class PdfDocument(models.Model):
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    file = models.FileField(upload_to="documents/")
    faiss_index = models.FileField(upload_to="indexes/", null=True, blank=True)
    upload_date=models.DateTimeField()
    def __str__(self):
        return f'{self.user.username}-->{self.file.name}'

class Chat(models.Model):
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    document=models.ForeignKey(PdfDocument,on_delete=models.CASCADE)
    title=models.CharField(max_length=250)
    created_at=models.DateTimeField()
    def __str__(self):
        return f'{self.title}'

class Message(models.Model):
    chat_id=models.ForeignKey(Chat,on_delete=models.CASCADE)
    sender=models.CharField(max_length=10,choices= [
        ('User', 'User'),
        ('Bot', 'Bot'),
    ])
    content=models.TextField()
    timestamp=models.DateTimeField()
    def __str__(self):
        return f'{self.content[:20]}...'
    
class DocumentChunk(models.Model):
    document = models.ForeignKey(PdfDocument, on_delete=models.CASCADE, related_name="chunks")
    text = models.TextField()
    page_number = models.IntegerField()

    def __str__(self):
        return f"Doc {self.document.id} - Page {self.page_number}: {self.text[:50]}..."

    