from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend
from django.db.models import Q
import logging

logger = logging.getLogger(__name__)
User = get_user_model()

class EmailBackend(ModelBackend):
    """
    Custom authentication backend that allows users to log in using their email address.
    """
    def authenticate(self, request, username=None, password=None, **kwargs):
        logger.info(f"Attempting to authenticate user with email: {username}")
        
        try:
            user = User.objects.get(Q(email__iexact=username))
            logger.info(f"Found user: {user}")
        except User.DoesNotExist:
            logger.warning(f"No user found with email: {username}")
            User().set_password(password)
            return None
        except User.MultipleObjectsReturned:
            logger.error(f"Multiple users found with email: {username}")
            return None
        
        password_valid = user.check_password(password)
        can_authenticate = self.user_can_authenticate(user)
        
        logger.info(f"Password valid: {password_valid}, Can authenticate: {can_authenticate}")
        
        if password_valid and can_authenticate:
            logger.info(f"Authentication successful for user: {user}")
            return user
        else:
            logger.warning(f"Authentication failed for user: {user}")
            return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None