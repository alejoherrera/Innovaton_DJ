import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import gradio as gr

class AuthManager:
    def __init__(self):
        # Obtener códigos de acceso desde variable de entorno
        self.access_codes = os.getenv("ACCESO_CODIGO", "1111,2222,3333,4444,5424").split(",")
        self.unlimited_code = "5424"
        self.daily_limit = 2
        
        # Archivo para almacenar el uso diario
        self.usage_file = Path("daily_usage.json")
        self.authenticated_sessions = set()
        
    def load_daily_usage(self):
        """Cargar el uso diario desde archivo"""
        if not self.usage_file.exists():
            return {}
        
        try:
            with open(self.usage_file, 'r') as f:
                data = json.load(f)
            
            # Limpiar datos de días anteriores
            today = datetime.now().strftime("%Y-%m-%d")
            if today not in data:
                data = {today: {}}
            else:
                # Mantener solo los datos de hoy
                data = {today: data.get(today, {})}
                
            return data
        except:
            return {}
    
    def save_daily_usage(self, usage_data):
        """Guardar el uso diario en archivo"""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(usage_data, f)
        except:
            pass
    
    def validate_code(self, code):
        """Validar el código de acceso"""
        return code.strip() in self.access_codes
    
    def can_make_query(self, code):
        """Verificar si el código puede hacer consultas"""
        if not self.validate_code(code):
            return False, "Código de acceso inválido"
        
        # Código ilimitado
        if code.strip() == self.unlimited_code:
            return True, "Acceso ilimitado"
        
        # Verificar límite diario para otros códigos
        usage_data = self.load_daily_usage()
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in usage_data:
            usage_data[today] = {}
        
        current_usage = usage_data[today].get(code.strip(), 0)
        
        if current_usage >= self.daily_limit:
            return False, f"Límite diario alcanzado ({self.daily_limit} consultas)"
        
        return True, f"Consultas restantes: {self.daily_limit - current_usage}"
    
    def record_query(self, code):
        """Registrar una consulta realizada"""
        if code.strip() == self.unlimited_code:
            return  # No registrar para código ilimitado
        
        usage_data = self.load_daily_usage()
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in usage_data:
            usage_data[today] = {}
        
        usage_data[today][code.strip()] = usage_data[today].get(code.strip(), 0) + 1
        self.save_daily_usage(usage_data)
    
    def create_auth_interface(self):
        """Crear la interfaz de autenticación"""
        with gr.Column(visible=True) as auth_column:
            gr.Markdown("### 🔐 Acceso al Sistema")
            gr.Markdown("Ingrese su código de acceso para continuar:")
            
            code_input = gr.Textbox(
                label="Código de acceso",
                placeholder="Ingrese su código...",
                type="password",
                elem_id="auth_code"
            )
            
            auth_btn = gr.Button("Verificar Acceso", variant="primary")
            auth_status = gr.Markdown("", visible=False)
            
        return auth_column, code_input, auth_btn, auth_status
    
    def authenticate(self, code):
        """Función de autenticación que devuelve el estado de la UI"""
        can_access, message = self.can_make_query(code)
        
        if can_access:
            # Agregar a sesiones autenticadas
            self.authenticated_sessions.add(code.strip())
            
            return (
                gr.update(visible=False),  # Ocultar auth_column
                gr.update(visible=True),   # Mostrar main_content
                gr.update(value=f"✅ {message}", visible=True),  # auth_status
                code.strip()  # Devolver código para usar en las consultas
            )
        else:
            return (
                gr.update(visible=True),   # Mantener auth_column visible
                gr.update(visible=False),  # Mantener main_content oculto
                gr.update(value=f"❌ {message}", visible=True),  # auth_status
                None  # No hay código válido
            )
    
    def is_authenticated(self, code):
        """Verificar si el código está autenticado"""
        return code in self.authenticated_sessions
    
    def check_query_permission(self, code):
        """Verificar permisos antes de hacer una consulta"""
        if not self.is_authenticated(code):
            return False, "Sesión no autenticada"
        
        can_access, message = self.can_make_query(code)
        return can_access, message