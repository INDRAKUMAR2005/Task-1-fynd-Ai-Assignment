import google.generativeai as genai
import os

API_KEY = "AIzaSyCX3jp99dBSNKge8pk1Fh62CltS0rSiKr8"
genai.configure(api_key=API_KEY)

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error: {e}")
