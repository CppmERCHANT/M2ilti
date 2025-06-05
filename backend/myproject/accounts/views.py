from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.views import LoginView
from django.contrib import messages
from django.views.generic import CreateView
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import requests
import logging
from .forms import CustomLoginForm, CustomRegistrationForm, PreferencesForm
from .models import UserPreferences

logger = logging.getLogger(__name__)

ML_SERVICE_URL = "http://127.0.0.1:5000"

class CustomLoginView(LoginView):
    form_class = CustomLoginForm
    template_name = 'accounts/login.html'
    
    def get_success_url(self):
        try:
            UserPreferences.objects.get(user=self.request.user)
            return reverse_lazy('dashboard')
        except UserPreferences.DoesNotExist:
            return reverse_lazy('preferences_setup')
    
    def form_valid(self, form):
        messages.success(self.request, 'Connexion réussie!')
        return super().form_valid(form)
    
    def form_invalid(self, form):
        messages.error(self.request, 'Email ou mot de passe incorrect.')
        return super().form_invalid(form)

class CustomRegistrationView(CreateView):
    form_class = CustomRegistrationForm
    template_name = 'accounts/register.html'
    success_url = reverse_lazy('login')

    def form_valid(self, form):
        try:
            user = form.save()
            messages.success(self.request, 'Inscription réussie! Vous pouvez maintenant vous connecter.')
            return redirect(self.success_url)
        except Exception as e:
            messages.error(self.request, f'Erreur lors de l\'inscription: {str(e)}')
            return self.form_invalid(form)
    
    def form_invalid(self, form):
        for field, errors in form.errors.items():
            for error in errors:
                messages.error(self.request, f'{field}: {error}')
        return super().form_invalid(form)

@login_required
def preferences_setup(request):
    try:
        preferences = UserPreferences.objects.get(user=request.user)
        return redirect('dashboard')
    except UserPreferences.DoesNotExist:
        pass
    
    if request.method == 'POST':
        form = PreferencesForm(request.POST)
        if form.is_valid():
            preferences = form.save(commit=False)
            preferences.user = request.user
            preferences.save()
            messages.success(request, 'Vos préférences ont été enregistrées avec succès!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Veuillez corriger les erreurs dans le formulaire.')
    else:
        form = PreferencesForm()
    
    return render(request, 'accounts/preferences_setup.html', {'form': form})

@login_required
def preferences_edit(request):
    try:
        preferences = UserPreferences.objects.get(user=request.user)
    except UserPreferences.DoesNotExist:
        return redirect('preferences_setup')
    
    if request.method == 'POST':
        form = PreferencesForm(request.POST, instance=preferences)
        if form.is_valid():
            form.save()
            messages.success(request, 'Vos préférences ont été mises à jour avec succès!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Veuillez corriger les erreurs dans le formulaire.')
    else:
        form = PreferencesForm(instance=preferences)
    
    return render(request, 'accounts/preferences_edit.html', {'form': form, 'preferences': preferences})

@login_required
def profile_view(request):
    try:
        preferences = UserPreferences.objects.get(user=request.user)
    except UserPreferences.DoesNotExist:
        preferences = None
    
    context = {
        'user': request.user,
        'preferences': preferences,
    }
    return render(request, 'accounts/profile.html', context)

@login_required
def my_choices(request):
    try:
        preferences = UserPreferences.objects.get(user=request.user)
    except UserPreferences.DoesNotExist:
        return redirect('preferences_setup')
    
    context = {
        'user': request.user,
        'preferences': preferences,
    }
    return render(request, 'accounts/my_choices.html', context)

def get_ml_service_status():
    """Check if ML service is running"""
    try:
        response = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_recommendations_from_ml(user_text):
    """Get recommendations from ML service"""
    try:
        response = requests.post(
            f"{ML_SERVICE_URL}/predict",
            data={'review': user_text},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"ML service error: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error calling ML service: {e}")
        return None

@login_required
def recommendations(request):
    try:
        preferences = UserPreferences.objects.get(user=request.user)
    except UserPreferences.DoesNotExist:
        return redirect('preferences_setup')
    
    # Check ML service status
    ml_status = get_ml_service_status()
    recommendations_data = None
    
    if request.method == 'POST':
        user_input = request.POST.get('user_input', '').strip()
        if user_input and ml_status:
            recommendations_data = get_recommendations_from_ml(user_input)
    
    context = {
        'user': request.user,
        'preferences': preferences,
        'ml_status': ml_status,
        'recommendations_data': recommendations_data,
    }
    return render(request, 'accounts/recommendations.html', context)

@csrf_exempt
@login_required
def get_recommendations_ajax(request):
    """AJAX endpoint for getting recommendations"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_text = data.get('text', '').strip()
            
            if not user_text:
                return JsonResponse({'error': 'Aucun texte fourni'}, status=400)
            
            # Get recommendations from ML service
            ml_response = get_recommendations_from_ml(user_text)
            
            if ml_response:
                return JsonResponse(ml_response)
            else:
                return JsonResponse({'error': 'Service de recommandation indisponible'}, status=503)
                
        except Exception as e:
            logger.error(f"Error in AJAX recommendations: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Méthode non autorisée'}, status=405)

@csrf_exempt
def process_id_card(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            
            extracted_data = {
                'nom': 'BENCHEIKH',
                'prenom': 'Ahmed',
                'numero_national': '1234567890123456',
                'success': True
            }
            
            return JsonResponse(extracted_data)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Méthode non autorisée'})

def custom_logout(request):
    logout(request)
    messages.success(request, 'Déconnexion réussie!')
    return redirect('login')

@login_required
def dashboard(request):
    try:
        preferences = UserPreferences.objects.get(user=request.user)
    except UserPreferences.DoesNotExist:
        return redirect('preferences_setup')
    
    # Check ML service status for dashboard
    ml_status = get_ml_service_status()
    
    context = {
        'user': request.user,
        'preferences': preferences,
        'ml_status': ml_status,
    }
    return render(request, 'accounts/dashboard.html', context)