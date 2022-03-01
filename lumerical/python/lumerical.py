import sys, os

sys.path.append("C:\\Program Files\\Lumerical\\v212\\api\\python\\") # Windows
sys.path.append("/opt/lumerical/v221/api/python/lumapi.py") # Linux


import lumapi


def get_api():
    return lumapi
