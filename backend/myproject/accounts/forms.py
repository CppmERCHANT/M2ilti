# forms.py - Create this file if it doesn't exist

from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import get_user_model
from .models import UserPreferences

User = get_user_model()

class CustomLoginForm(AuthenticationForm):
    username = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Email'
        }),
        label='Email'
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Mot de passe'
        }),
        label='Mot de passe'
    )

class CustomRegistrationForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Email'
        }),
        label='Email'
    )
    nom = forms.CharField(
        max_length=100,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nom'
        }),
        label='Nom'
    )
    prenom = forms.CharField(
        max_length=100,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Prénom'
        }),
        label='Prénom'
    )
    numero_national = forms.CharField(
        max_length=20,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Numéro National'
        }),
        label='Numéro National'
    )
    carte_identite = forms.ImageField(
        required=False,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*'
        }),
        label='Carte d\'identité'
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Mot de passe'
        }),
        label='Mot de passe'
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirmer le mot de passe'
        }),
        label='Confirmer le mot de passe'
    )

    class Meta:
        model = User
        fields = ('email', 'nom', 'prenom', 'numero_national', 'carte_identite', 'password1', 'password2')

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('Un utilisateur avec cet email existe déjà.')
        return email

    def clean_numero_national(self):
        numero_national = self.cleaned_data.get('numero_national')
        if User.objects.filter(numero_national=numero_national).exists():
            raise forms.ValidationError('Un utilisateur avec ce numéro national existe déjà.')
        return numero_national

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.nom = self.cleaned_data['nom']
        user.prenom = self.cleaned_data['prenom']
        user.numero_national = self.cleaned_data['numero_national']
        
        if commit:
            user.save()
            if self.cleaned_data.get('carte_identite'):
                user.carte_identite = self.cleaned_data['carte_identite']
                user.save()
        
        return user


class PreferencesForm(forms.ModelForm):
    class Meta:
        model = UserPreferences
        fields = ['domaine_etude', 'type_diplome', 'interet']
        widgets = {
            'domaine_etude': forms.Select(attrs={
                'class': 'form-control',
            }),
            'type_diplome': forms.Select(attrs={
                'class': 'form-control',
            }),
            'interet': forms.Select(attrs={
                'class': 'form-control',
            }),
        }
        labels = {
            'domaine_etude': "Domaine d'étude",
            'type_diplome': 'Type de diplôme',
            'interet': 'Intérêt professionnel',
        }