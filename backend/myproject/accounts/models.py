from django.contrib.auth.models import AbstractUser
from django.contrib.auth.base_user import BaseUserManager
from django.db import models

class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('Le email doit être renseigné')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self.create_user(email, password, **extra_fields)


class CustomUser(AbstractUser):
    username = None
    email = models.EmailField(unique=True, verbose_name="Email")
    nom = models.CharField(max_length=100, verbose_name="Nom")
    prenom = models.CharField(max_length=100, verbose_name="Prénom")
    numero_national = models.CharField(max_length=20, unique=True, verbose_name="Numéro National")
    carte_identite = models.ImageField(upload_to='cartes_identite/', null=True, blank=True, verbose_name="Carte d'identité")
    date_creation = models.DateTimeField(auto_now_add=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['nom', 'prenom', 'numero_national']
    objects = CustomUserManager()

    class Meta:
        verbose_name = "Utilisateur"
        verbose_name_plural = "Utilisateurs"

    def __str__(self):
        return f"{self.prenom} {self.nom} ({self.email})"


class UserPreferences(models.Model):
    DOMAINE_CHOICES = [
        ('informatique', 'Informatique'),
        ('medecine', 'Médecine'),
        ('ingenierie', 'Ingénierie'),
        ('economie', 'Économie'),
        ('droit', 'Droit'),
        ('lettres', 'Lettres et Langues'),
        ('sciences', 'Sciences Exactes'),
        ('sciences_sociales', 'Sciences Sociales'),
        ('arts', 'Arts et Design'),
        ('agriculture', 'Agriculture'),
    ]
    
    TYPE_DIPLOME_CHOICES = [
        ('licence', 'Licence'),
        ('master', 'Master'),
        ('doctorat', 'Doctorat'),
        ('ingenieur', 'Ingénieur'),
        ('technicien_superieur', 'Technicien Supérieur'),
        ('formation_professionnelle', 'Formation Professionnelle'),
    ]
    
    INTERET_CHOICES = [
        ('recherche', 'Recherche Académique'),
        ('industrie', 'Secteur Industriel'),
        ('enseignement', 'Enseignement'),
        ('entrepreneuriat', 'Entrepreneuriat'),
        ('fonction_publique', 'Fonction Publique'),
        ('secteur_prive', 'Secteur Privé'),
        ('consulting', 'Conseil'),
        ('international', 'Organisations Internationales'),
    ]
    
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='preferences')
    domaine_etude = models.CharField(max_length=50, choices=DOMAINE_CHOICES, verbose_name="Domaine d'étude")
    type_diplome = models.CharField(max_length=50, choices=TYPE_DIPLOME_CHOICES, verbose_name="Type de diplôme")
    interet = models.CharField(max_length=50, choices=INTERET_CHOICES, verbose_name="Intérêt professionnel")
    date_creation = models.DateTimeField(auto_now_add=True)
    date_modification = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Préférence utilisateur"
        verbose_name_plural = "Préférences utilisateurs"
    
    def __str__(self):
        return f"Préférences de {self.user.prenom} {self.user.nom}"