from django.db import models
from django.core.exceptions import ValidationError
from django.contrib.auth.models import User


# Create your models here.
class Data(models.Model):
    Employee_ID = models.CharField(max_length=10)
    Domain = models.CharField(max_length=50)
    Other_domain = models.CharField(max_length=50, blank=True, null=True)
    Working_years = models.CharField(max_length=50)
    Bored = models.CharField(max_length=10)
    Free_time = models.CharField(max_length=100, null=True, blank=True)
    Satisfied_with_company = models.CharField(max_length=10)
    Improve_company = models.TextField(blank=True, null=True)
    Recommend_friends = models.CharField(max_length=10)
    Working_team = models.CharField(max_length=10)
    Team_improve = models.TextField(blank=True, null=True)
    Coming_to_work = models.TextField(blank=True, null=True)
    Satisfied_with_manager = models.CharField(max_length=10)
    Manager_improve = models.TextField(blank=True, null=True)
    Culture_Values = models.CharField(max_length=10)
    Compensation_Benefits = models.CharField(max_length=10)
    Satisfied_with_management = models.CharField(max_length=10)
    Management_improve = models.TextField(blank=True, null=True)
    Improve_work = models.TextField(blank=True, null=True)
    Satisfied_with_HR = models.CharField(max_length=10)
    Hr_improve = models.TextField(blank=True, null=True)
    Work_life_balance = models.CharField(max_length=10)
    Suggestions = models.TextField(blank=True, null=True)


def validate_mail(value):
    if "@ratnaglobaltech.com" in value:
        return value
    else:
        raise ValidationError("Only ratnaglobaltech.com domain will accept")


class Employee(models.Model):
    Employee_ID = models.CharField(max_length=10)
    Employee_Firstname = models.CharField(max_length=100)
    Employee_Lastname = models.CharField(max_length=100)
    Employee_Domain = models.CharField(max_length=100)
    Employee_Mail = models.CharField(max_length=50, validators=[validate_mail])
    Employee_Password = models.CharField(max_length=50)


class Otp(models.Model):
    Employee_Mail = models.CharField(max_length=50, validators=[validate_mail])
    Employee_Secret_key = models.CharField(max_length=100)
    Time = models.DateTimeField(null=True)
    Otp = models.CharField(max_length=10)


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    totp_secret_key = models.CharField(max_length=64)
