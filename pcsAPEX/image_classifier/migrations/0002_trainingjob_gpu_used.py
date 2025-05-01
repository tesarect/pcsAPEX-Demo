from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image_classifier', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingjob',
            name='gpu_used',
            field=models.BooleanField(default=False, help_text='Whether GPU was used for training'),
        ),
    ]
