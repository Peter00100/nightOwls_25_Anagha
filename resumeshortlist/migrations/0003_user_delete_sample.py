# Generated by Django 4.2.4 on 2023-12-22 09:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("resumeshortlist", "0002_rename_user_sample"),
    ]

    operations = [
        migrations.CreateModel(
            name="user",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=75)),
                ("dob", models.CharField(max_length=75)),
                ("sex", models.CharField(max_length=75)),
                ("email", models.CharField(max_length=75)),
                ("contactno", models.CharField(max_length=75)),
                ("username", models.CharField(max_length=75)),
                ("password", models.CharField(max_length=75)),
                ("address", models.CharField(max_length=75)),
                ("selected", models.IntegerField()),
            ],
        ),
        migrations.DeleteModel(
            name="Sample",
        ),
    ]