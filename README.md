# Insider Threat Detection System

A web-based insider threat detection platform built to identify suspicious employee/user behavior through anomaly detection and behavioral monitoring.

## Features

- User activity monitoring dashboard
- Insider threat risk scoring
- Behavioral anomaly detection
- Admin panel for threat review
- Data visualization dashboards
- Role-based authentication system

## Tech Stack

- Python
- Django
- SQLite
- HTML/CSS/Bootstrap
- Machine Learning / Anomaly Detection

## Project Structure

admins/ - Admin management module  
users/ - User monitoring and auth module  
templates/ - Frontend templates  
static/ - Static assets  
threat_detection/ - Core project configuration  

## Installation

```bash
git clone https://github.com/saisreeja6905/insider-threat-detection.git
cd insider-threat-detection
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
