import gradio as gr
import os
import tempfile
from pathlib import Path
from datetime import datetime
from document_analyzer import DocumentAnalyzer

def create_analysis_interface(gemini_api_key: str, qdrant_url: str, qdrant_api_key: str):
    """Crea la interfaz de análisis de documentos"""
    
    # Inicializar el analizador
    analyzer = DocumentAnalyzer(gemini_api_key, qdrant_url, qdrant_api_key)
    
    def analyze_uploaded_document(file, progress=gr.Progress()):
        """Analiza el documento subido"""
        if file is None:
            return "❌ Error: No se ha subido ningún archivo", "", "", gr.update(visible=False), gr.update(visible=False)
        
        try:
            # Mostrar progreso inicial
            progress(0.1, desc="Extrayendo texto del PDF...ya casi EUREKA")
            
            # Analizar el documento
            result = analyzer.analyze_document(file.name)
            
            if not result['success']:
                return f"❌ Error: {result['error']}", "", "", gr.update(visible=False), gr.update(visible=False)
            
            # Progreso de generación de resúmenes
            progress(0.7, desc="Generando resúmenes con IA...ya casi EUREKA")
            
            # Simular progreso gradual
            import time
            time.sleep(1)
            
            progress(1.0, desc="¡EUREKA! Análisis completado")
            
            status_final = "✅ **Documento analizado exitosamente!** \n\n✏️ Puedes editar los resúmenes si lo deseas antes de buscar precedentes."
            facts = result['facts_summary']
            legal = result['legal_summary']
            
            # Mostrar botón de análisis de precedentes y área de precedentes
            return status_final, facts, legal, gr.update(visible=True), gr.update(visible=True)
            
        except Exception as e:
            return f"❌ Error inesperado: {str(e)}", "", "", gr.update(visible=False), gr.update(visible=False)
    
    def search_precedents_action(file, facts_text, legal_text, progress=gr.Progress()):
        """Busca precedentes relacionados basándose en argumentos jurídicos"""
        if file is None:
            return "❌ Error: No hay documento para analizar precedentes", gr.update(visible=False)
        
        if not legal_text or legal_text.strip() == "":
            return "❌ Error: Primero debe generar los argumentos jurídicos para buscar precedentes", gr.update(visible=False)
        
        try:
            # Progreso inicial
            progress(0.1, desc="Generando embeddings de argumentos jurídicos...")
            
            import time
            time.sleep(1)
            
            # Progreso búsqueda
            progress(0.4, desc="Buscando en base de datos vectorial...")
            time.sleep(1)
            
            # Progreso filtrado IA
            progress(0.7, desc="Filtrado inteligente con IA...ya casi EUREKA")
            
            # Buscar precedentes basándose en argumentos jurídicos
            precedents = analyzer.search_precedents(legal_text, limit=20)
            
            progress(1.0, desc="¡EUREKA! Precedentes analizados")
            
            if not precedents:
                precedents_html = """
                <div class="precedents-container">
                    <h3>📚 PRECEDENTES RELACIONADOS</h3>
                    <p><strong>✅ Análisis completado</strong></p>
                    <p><strong>Criterio de búsqueda:</strong> Basado en argumentos jurídicos únicamente</p>
                    <p><strong>Filtro inteligente:</strong> El modelo evalúa y muestra solo precedentes con relación jurídica real (Alta, Media o Baja)</p>
                    <p><em>❌ No se encontraron precedentes con relación jurídica significativa.</em></p>
                </div>
                """
            else:
                precedents_html = """
                <div class="precedents-container">
                    <h3>📚 PRECEDENTES RELACIONADOS</h3>
                    <p><strong>✅ Análisis completado</strong></p>
                    <p><strong>Criterio de búsqueda:</strong> Basado en argumentos jurídicos únicamente para mayor precisión</p>
                    <p><strong>Filtro inteligente:</strong> El modelo evalúa y muestra solo precedentes con relación jurídica real</p>
                    <hr>
                """
                
                for i, precedent in enumerate(precedents, 1):
                    # Truncar contenido para mejor legibilidad
                    content_preview = precedent['document'][:200].replace('\n', ' ')
                    if len(precedent['document']) > 200:
                        content_preview += "..."
                    
                    relation_level = precedent.get('relation_level', 'NO DETERMINADO')
                    relation_justification = precedent.get('relation_justification', 'Justificación no disponible')
                    
                    # Color según nivel de relación
                    level_color = {
                        'ALTA': '#28a745',  # Verde
                        'MEDIA': '#ffc107',  # Amarillo
                        'BAJA': '#fd7e14'   # Naranja
                    }.get(relation_level, '#6c757d')
                    
                    precedents_html += f"""
                    <div class="precedent-item">
                        <h4>🔍 Precedente {i}</h4>
                        <p><strong>📊 Nivel de relación:</strong> <span class="relation-level" style="background-color: {level_color}; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;">{relation_level}</span></p>
                        <p><strong>📄 Fuente:</strong> {precedent['source']}</p>
                        <p><strong>🔗 Justificación de la relación (argumento/precedente):</strong></p>
                        <p style="font-style: italic; background-color: #f8f9fa; padding: 10px; border-left: 4px solid #007bff; margin: 10px 0;"><em>{relation_justification}</em></p>
                        <p><strong>📝 Contenido:</strong> <em>{content_preview}</em></p>
                    </div>
                    <hr>
                    """
                
                precedents_html += "</div>"
            
            return precedents_html, gr.update(visible=True)
            
        except Exception as e:
            error_html = f"""
            <div class="precedents-container">
                <h3>❌ ERROR EN ANÁLISIS</h3>
                <p><strong>Error:</strong> {str(e)}</p>
            </div>
            """
            return error_html, gr.update(visible=False)
    
    def generate_pdf_report(file, facts_text, legal_text, precedents_text, progress=gr.Progress()):
        """Genera y descarga el reporte PDF"""
        if file is None:
            return None, "❌ Error: No hay documento para generar reporte"
        
        try:
            # Progreso inicial
            progress(0.1, desc="Recopilando información del análisis...")
            
            # Obtener nombre del documento original
            document_name = Path(file.name).name
            
            # Crear nombre personalizado para el PDF
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"EUREKA_report_{timestamp}.pdf"
            
            # Crear archivo temporal con el nombre personalizado
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, pdf_filename)
            
            progress(0.3, desc="Procesando precedentes encontrados...")
            
            # Buscar precedentes si no están disponibles
            precedents = []
            if precedents_text and "PRECEDENTES RELACIONADOS" in precedents_text:
                # Extraer precedentes basándose en argumentos jurídicos (ya filtrados por el modelo)
                precedents = analyzer.search_precedents(legal_text, limit=20)
            
            progress(0.6, desc="Generando documento PDF...")
            
            # Prompts utilizados
            facts_prompt = """Analiza el documento y genera un resumen CONCISO de hechos y personas en MÁXIMO 3 PÁRRAFOS:
PÁRRAFO 1 - HECHOS CRONOLÓGICOS (máximo 6 líneas): Principales hechos en orden cronológico con fechas.
PÁRRAFO 2 - PERSONAS FISCALIZADAS (máximo 4 líneas): Nombres, cargos y roles de involucrados.
PÁRRAFO 3 - DATOS CLAVE (máximo 4 líneas): Montos económicos, instituciones fiscalizadas y lugares relevantes."""
            
            legal_prompt = """Analiza el documento y extrae los 5 PRINCIPALES ARGUMENTOS JURÍDICOS más relevantes.
Cada argumento debe tener: título y resumen de exactamente 5 líneas explicando normativa citada, 
interpretación legal y conclusión. Enfócate en argumentos jurídicos sólidos con leyes, artículos, 
principios jurídicos y conclusiones legales."""
            
            progress(0.8, desc="Finalizando reporte...")
            
            # Generar PDF con análisis de relación
            success = analyzer.generate_pdf_report(
                document_name=document_name,
                facts_summary=facts_text,
                legal_summary=legal_text,
                precedents=precedents,
                facts_prompt=facts_prompt,
                legal_prompt=legal_prompt,
                output_path=output_path,
                search_note="La búsqueda de precedentes se realizó basándose únicamente en los argumentos jurídicos para mayor precisión."
            )
            
            progress(1.0, desc="¡EUREKA! Reporte PDF completado")
            
            if success:
                return output_path, f"✅ Reporte PDF generado exitosamente: {pdf_filename}"
            else:
                return None, "❌ Error al generar el reporte PDF"
                
        except Exception as e:
            return None, f"❌ Error inesperado: {str(e)}"
    
    # Crear la interfaz
    with gr.Blocks(css="""
        .analysis-container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .status-box { padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #007bff; background-color: #f8f9fa; }
        .success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; border-left: 4px solid #28a745; }
        .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; border-left: 4px solid #dc3545; }
        .processing { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; border-left: 4px solid #ffc107; }
        .summary-box { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0; }
        .summary-textbox textarea { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            line-height: 1.5 !important;
        }
        .precedents-container {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .precedents-container h3 {
            color: #284293;
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 18px;
            font-weight: bold;
        }
        .precedents-container h4 {
            color: #495057;
            margin-top: 15px;
            margin-bottom: 10px;
            font-size: 16px;
            font-weight: bold;
        }
        .precedent-item {
            background-color: white;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        .precedent-item:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        .score {
            background-color: #007bff;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .precedents-container hr {
            border: none;
            height: 1px;
            background-color: #dee2e6;
            margin: 15px 0;
        }
        .progress-indicator {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #007bff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    """) as analysis_interface:
        
        gr.Markdown("# 📄 Análisis de Documentos")
        gr.Markdown("Sube un informe de fiscalización de la Contraloría General u otro documento similar para su análisis.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Upload del archivo
                file_upload = gr.File(
                    label="📎 Subir documento PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                # Botón de análisis
                analyze_btn = gr.Button("🔍 Analizar Documento", variant="primary", size="lg")
                
                # Estado del análisis
                status_text = gr.Markdown("")
        
        # Resúmenes editables
        with gr.Row():
            with gr.Column():
                facts_summary = gr.Textbox(
                    label="👥 Resumen de Hechos y Personas",
                    placeholder="El resumen de hechos y personas aparecerá aquí...",
                    lines=15,
                    interactive=True,
                    elem_classes=["summary-textbox"]
                )
            
            with gr.Column():
                legal_summary = gr.Textbox(
                    label="⚖️ Principales Argumentos Jurídicos",
                    placeholder="Los argumentos jurídicos aparecerán aquí...",
                    lines=15,
                    interactive=True,
                    elem_classes=["summary-textbox"]
                )
        
        # Botón para analizar precedentes
        with gr.Row():
            precedents_btn = gr.Button(
                "🔍 Analizar Precedentes Jurídicos (Filtro Inteligente - 2-5 min)", 
                variant="secondary", 
                size="lg",
                visible=False
            )
        
        # Área de precedentes
        precedents_area = gr.HTML(
            label="📚 Precedentes Relacionados",
            visible=False
        )
        
        # Botón de descarga
        with gr.Row():
            with gr.Column():
                download_btn = gr.Button("📥 Generar Reporte PDF", variant="primary", size="lg")
            with gr.Column():
                download_status = gr.Markdown("")
                pdf_output = gr.File(label="Descargar Reporte", visible=False)
        
        # Eventos con Progress tracking
        analyze_btn.click(
            analyze_uploaded_document,
            inputs=[file_upload],
            outputs=[status_text, facts_summary, legal_summary, precedents_btn, precedents_area],
            show_progress="full"  # Agregar para mostrar la barra de progreso
        )
        
        precedents_btn.click(
            search_precedents_action,
            inputs=[file_upload, facts_summary, legal_summary],
            outputs=[precedents_area, precedents_area],
            show_progress="full"  # Agregar para mostrar la barra de progreso
        )
        
        download_btn.click(
            generate_pdf_report,
            inputs=[file_upload, facts_summary, legal_summary, precedents_area],
            outputs=[pdf_output, download_status],
            show_progress="full"  # Agregar para mostrar la barra de progreso
        ).then(
            lambda file, status: (gr.update(visible=file is not None), status) if file else (gr.update(visible=False), status),
            inputs=[pdf_output, download_status],
            outputs=[pdf_output, download_status]
        )
    
    return analysis_interface

def create_analysis_tab(gemini_api_key: str, qdrant_url: str, qdrant_api_key: str):
    """Función helper para crear solo la pestaña de análisis"""
    return create_analysis_interface(gemini_api_key, qdrant_url, qdrant_api_key)