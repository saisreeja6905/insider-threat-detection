from django import forms
from .models import User_Registration
import re

class UserRegistrationForm(forms.ModelForm):
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': "Don't use special characters"})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': "Don't use special characters"})
    )

    class Meta:
        model = User_Registration
        fields = ['name', 'username', 'email', 'phone_number', 'address', 'password', 'confirm_password']

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        # Check if passwords match
        if password != confirm_password:
            raise forms.ValidationError("Passwords do not match.")

        # Validate password strength (example: minimum 8 characters, one letter, one number)
        pattern = r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}$'
        if not re.match(pattern, password):
            raise forms.ValidationError("Password must be at least 8 characters long and include a letter and a number.")

        return cleaned_data

    def clean_phone_number(self):
        phone = self.cleaned_data.get('phone_number')
        if not re.match(r'^\+?\d{10,15}$', phone):
            raise forms.ValidationError("Enter a valid phone number.")
        return phone
