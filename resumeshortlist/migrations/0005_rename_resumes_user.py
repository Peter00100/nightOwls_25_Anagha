# Generated by Django 4.2.4 on 2023-12-22 10:01

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("resumeshortlist", "0004_resumes_delete_user"),
    ]

    operations = [
        migrations.RenameModel(
            old_name="resumes",
            new_name="user",
        ),
    ]
