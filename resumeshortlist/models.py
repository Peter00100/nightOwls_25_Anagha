from django.db import models

# Create your models here.

class User(models.Model):
    sno = models.AutoField(primary_key=True)
    name = models.CharField(max_length=75)
    dob= models.CharField(max_length=75)
    sex = models.CharField(max_length=75)
    email = models.CharField(max_length=75)
    contactno = models.CharField(max_length=75)
    username = models.CharField(max_length=75)
    password = models.CharField(max_length=75)
    address = models.CharField(max_length=75)
    selected = models.IntegerField(default=2)


# class manjil(models.Model):
#     no = models.AutoField(primary_key=True)
#     name = models.CharField(max_length=75)