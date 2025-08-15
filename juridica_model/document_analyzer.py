import os
import uuid
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

class DocumentAnalyzer:
    def __init__(self, gemini_api_key: str, qdrant_url: str, qdrant_api_key: str, collection_name: str = "resoluciones"):
        """
        Inicializa el analizador de documentos
        """
        self.gemini_api_key = gemini_api_key
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        
        # Configurar Gemini
        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel('gemini-1.5-flash')
        
        # Configurar Qdrant
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrae texto de un archivo PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Error al extraer texto del PDF: {str(e)}")
    
    def generate_facts_summary(self, document_text: str) -> str:
        """Genera resumen conciso de hechos y personas usando Gemini"""
        prompt_facts = """
        Analiza el siguiente documento y genera un resumen CONCISO de hechos y personas en MÁXIMO 3 PÁRRAFOS.
        
        ESTRUCTURA REQUERIDA:
        
        PÁRRAFO 1 - HECHOS CRONOLÓGICOS (máximo 6 líneas):
        Describe los principales hechos en orden cronológico con fechas específicas.
        
        PÁRRAFO 2 - PERSONAS FISCALIZADAS (máximo 4 líneas):
        Lista nombres completos, cargos y roles de las personas involucradas.
        
        PÁRRAFO 3 - DATOS CLAVE (máximo 4 líneas):
        - Montos económicos involucrados
        - Instituciones fiscalizadas  
        - Lugares geográficos relevantes
        
        IMPORTANTE: Usa texto plano, sin formato markdown. Máximo 3 párrafos total.
        
        Documento:
        {document_text}
        """
        
        try:
            # Para documentos largos, usar muestra representativa
            if len(document_text) > 50000:
                # Tomar inicio (primera parte), desarrollo (medio) y conclusiones (final)
                start_chunk = document_text[:15000]
                middle_start = len(document_text) // 2 - 7500
                middle_chunk = document_text[middle_start:middle_start + 15000]
                end_chunk = document_text[-15000:]
                
                sample_text = f"{start_chunk}\n\n{middle_chunk}\n\n{end_chunk}"
                document_text = sample_text
            
            response = self.llm.generate_content(prompt_facts.format(document_text=document_text))
            return response.text
        except Exception as e:
            return f"Error al generar resumen de hechos: {str(e)}"
    
    def generate_legal_summary(self, document_text: str) -> str:
        """Genera resumen de los 5 principales argumentos jurídicos usando Gemini"""
        prompt_legal = """
        Analiza el siguiente documento y extrae los 5 PRINCIPALES ARGUMENTOS JURÍDICOS más relevantes.
        
        FORMATO REQUERIDO:
        
        ARGUMENTO 1: [Título del argumento]
        [Resumen de exactamente 5 líneas explicando este argumento jurídico, incluyendo normativa citada, interpretación legal y conclusión]
        
        ARGUMENTO 2: [Título del argumento]
        [Resumen de exactamente 5 líneas explicando este argumento jurídico, incluyendo normativa citada, interpretación legal y conclusión]
        
        ARGUMENTO 3: [Título del argumento]
        [Resumen de exactamente 5 líneas explicando este argumento jurídico, incluyendo normativa citada, interpretación legal y conclusión]
        
        ARGUMENTO 4: [Título del argumento]
        [Resumen de exactamente 5 líneas explicando este argumento jurídico, incluyendo normativa citada, interpretación legal y conclusión]
        
        ARGUMENTO 5: [Título del argumento]
        [Resumen de exactamente 5 líneas explicando este argumento jurídico, incluyendo normativa citada, interpretación legal y conclusión]
        
        IMPORTANTE: 
        - Usa texto plano, sin formato markdown
        - Cada argumento debe tener exactamente 5 líneas de descripción
        - Enfócate en los argumentos jurídicos más sólidos y relevantes
        - Incluye leyes, artículos, principios jurídicos y conclusiones legales
        
        Documento:
        {document_text}
        """
        
        try:
            # Para documentos largos, usar muestra representativa enfocada en aspectos legales
            if len(document_text) > 50000:
                # Buscar secciones que contengan términos jurídicos
                legal_terms = ['artículo', 'ley', 'reglamento', 'decreto', 'disposición', 'conclusión', 'normativa']
                
                # Tomar inicio, secciones legales y conclusiones
                start_chunk = document_text[:15000]
                
                # Buscar sección de conclusiones/disposiciones (usualmente al final)
                end_chunk = document_text[-20000:]
                
                # Buscar secciones con alta densidad de términos legales
                middle_start = len(document_text) // 2 - 10000
                middle_chunk = document_text[middle_start:middle_start + 20000]
                
                sample_text = f"{start_chunk}\n\n{middle_chunk}\n\n{end_chunk}"
                document_text = sample_text
            
            response = self.llm.generate_content(prompt_legal.format(document_text=document_text))
            return response.text
        except Exception as e:
            return f"Error al generar resumen jurídico: {str(e)}"
    
    def chunk_text(self, text: str, max_chars: int = 30000, overlap: int = 500) -> List[str]:
        """Divide el texto en chunks más pequeños para embeddings"""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            # Si no es el último chunk, buscar un punto de corte natural
            if end < len(text):
                # Buscar el último punto, salto de línea o espacio antes del límite
                for delimiter in ['\n\n', '\n', '. ', ' ']:
                    last_delimiter = text.rfind(delimiter, start, end)
                    if last_delimiter > start:
                        end = last_delimiter + len(delimiter)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Calcular el siguiente inicio con overlap
            start = max(start + 1, end - overlap)
            
            # Evitar bucle infinito
            if start >= len(text):
                break
        
        return chunks
    
    def get_document_embedding(self, text: str) -> List[float]:
        """Genera embedding del documento usando Gemini con manejo de tamaño"""
        try:
            # Si el texto es muy largo, usar solo una muestra representativa
            if len(text) > 30000:
                # Tomar el inicio, medio y final del documento
                start_chunk = text[:10000]
                middle_start = len(text) // 2 - 5000
                middle_chunk = text[middle_start:middle_start + 10000]
                end_chunk = text[-10000:]
                
                # Combinar las muestras
                sample_text = f"{start_chunk}\n\n[...CONTENIDO INTERMEDIO...]\n\n{middle_chunk}\n\n[...CONTENIDO FINAL...]\n\n{end_chunk}"
                text = sample_text[:29000]  # Asegurar que esté bajo el límite
            
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generando embedding: {e}")
            # Como fallback, usar el primer chunk pequeño
            try:
                first_chunk = text[:20000]
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=first_chunk,
                    task_type="retrieval_query"
                )
                return result['embedding']
            except:
                print("Usando embedding por defecto debido a errores")
                return [0.0] * 768
    
    def search_precedents(self, legal_arguments: str, limit: int = 10) -> List[Dict]:
        """Busca precedentes relacionados basándose en argumentos jurídicos"""
        try:
            # Generar embedding basado solo en argumentos jurídicos
            query_embedding = self.get_document_embedding(legal_arguments)
            
            # Buscar en Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            precedents = []
            for result in search_results:
                # Analizar relación jurídica
                relation_analysis = self._analyze_legal_relation(legal_arguments, result.payload.get("document", ""))
                
                # Solo incluir precedentes con relación ALTA, MEDIA o BAJA (no NINGUNA)
                if relation_analysis["nivel"] in ["ALTA", "MEDIA", "BAJA"]:
                    precedent = {
                        'document': result.payload.get("document", ""),
                        'metadata': result.payload.get("metadata", {}),
                        'source': result.payload.get("metadata", {}).get("source", "Desconocido"),
                        'relation_level': relation_analysis["nivel"],
                        'relation_justification': relation_analysis["justificacion"]
                    }
                    precedents.append(precedent)
            
            return precedents
            
        except Exception as e:
            print(f"Error buscando precedentes: {e}")
            return []
    
    def _analyze_legal_relation(self, query_arguments: str, precedent_content: str) -> Dict[str, str]:
        """Analiza la relación jurídica entre los argumentos consultados y el precedente"""
        try:
            relation_prompt = f"""
            Analiza la relación jurídica entre los argumentos del documento consultado y este precedente.
            
            ARGUMENTOS DEL DOCUMENTO CONSULTADO:
            {query_arguments[:2000]}
            
            CONTENIDO DEL PRECEDENTE:
            {precedent_content[:2000]}
            
            Evalúa si existe una relación jurídica real y significativa. Responde en formato:
            
            NIVEL_RELACION: [ALTA/MEDIA/BAJA/NINGUNA]
            JUSTIFICACION: [Si hay relación (ALTA, MEDIA o BAJA), explica en 3-4 líneas detalladas: 
            1) Qué argumento específico se relaciona, 2) Cuál es la conexión jurídica concreta, 
            3) Qué normativa o principio legal los conecta. Si es NINGUNA, explica brevemente por qué no hay relación.]
            
            Sé estricto: solo marca como ALTA/MEDIA/BAJA si realmente existe una conexión jurídica sustancial.
            """
            
            response = self.llm.generate_content(relation_prompt)
            
            if response and response.text:
                text = response.text.strip()
                
                # Extraer nivel y justificación
                nivel = "NINGUNA"
                justificacion = "Análisis no disponible"
                
                if "NIVEL_RELACION:" in text:
                    lines = text.split('\n')
                    for line in lines:
                        if "NIVEL_RELACION:" in line:
                            nivel = line.split("NIVEL_RELACION:")[1].strip()
                        elif "JUSTIFICACION:" in line:
                            justificacion = line.split("JUSTIFICACION:")[1].strip()
                
                return {
                    "nivel": nivel,
                    "justificacion": justificacion
                }
            
            return {
                "nivel": "NINGUNA",
                "justificacion": "No se pudo analizar la relación"
            }
            
        except Exception as e:
            return {
                "nivel": "NINGUNA", 
                "justificacion": f"Error analizando relación: {str(e)}"
            }
    
    def generate_pdf_report(self, 
                          document_name: str,
                          facts_summary: str, 
                          legal_summary: str, 
                          precedents: List[Dict],
                          facts_prompt: str,
                          legal_prompt: str,
                          output_path: str,
                          search_note: str = "") -> bool:
        """Genera un reporte PDF con toda la información"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter,
                                  leftMargin=72, rightMargin=72, 
                                  topMargin=72, bottomMargin=72)
            styles = getSampleStyleSheet()
            
            # Estilos personalizados
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                textColor='darkblue',
                alignment=1,  # Centro
                spaceAfter=30
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor='darkblue',
                spaceBefore=20,
                spaceAfter=15
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=11,
                leading=16.5,  # Espacio y medio (11 * 1.5)
                alignment=4,   # Justificado
                spaceAfter=12
            )
            
            story = []
            
            # Título del reporte
            story.append(Paragraph("ANÁLISIS DE DOCUMENTO", title_style))
            story.append(Spacer(1, 20))
            
            # Información del documento
            story.append(Paragraph(f"<b>Documento analizado:</b> {document_name}", normal_style))
            story.append(Paragraph(f"<b>Fecha de análisis:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", normal_style))
            story.append(Paragraph(f"<b>Modelo LLM usado:</b> Gemini 1.5 Flash", normal_style))
            story.append(Paragraph(f"<b>Clúster Qdrant consultado:</b> {self.collection_name}", normal_style))
            if search_note:
                story.append(Paragraph(f"<b>Nota de búsqueda:</b> {search_note}", normal_style))
            story.append(Spacer(1, 30))
            
            # Prompts utilizados
            story.append(Paragraph("PROMPTS UTILIZADOS", heading_style))
            
            story.append(Paragraph("<b>Prompt para hechos y personas:</b>", normal_style))
            story.append(Paragraph(facts_prompt, normal_style))
            story.append(Spacer(1, 15))
            
            story.append(Paragraph("<b>Prompt para argumentos jurídicos:</b>", normal_style))
            story.append(Paragraph(legal_prompt, normal_style))
            story.append(Spacer(1, 30))
            
            # Resumen de hechos y personas
            story.append(Paragraph("RESUMEN DE HECHOS Y PERSONAS", heading_style))
            story.append(Paragraph(facts_summary, normal_style))
            story.append(Spacer(1, 30))
            
            # Resumen de argumentos jurídicos
            story.append(Paragraph("PRINCIPALES ARGUMENTOS JURÍDICOS", heading_style))
            story.append(Paragraph(legal_summary, normal_style))
            story.append(Spacer(1, 30))
            
            # Precedentes encontrados
            story.append(Paragraph("PRECEDENTES RELACIONADOS", heading_style))
            
            if precedents:
                for i, precedent in enumerate(precedents, 1):
                    relation_level = precedent.get('relation_level', 'NO DETERMINADO')
                    relation_justification = precedent.get('relation_justification', 'Justificación no disponible')
                    
                    story.append(Paragraph(f"<b>Precedente {i} - Nivel de relación: {relation_level}</b>", normal_style))
                    story.append(Paragraph(f"<b>Fuente:</b> {precedent['source']}", normal_style))
                    story.append(Paragraph(f"<b>Justificación de la relación (argumento/precedente):</b> {relation_justification}", normal_style))
                    story.append(Paragraph(f"<b>Contenido:</b> {precedent['document'][:500]}...", normal_style))
                    story.append(Spacer(1, 20))
            else:
                story.append(Paragraph("No se encontraron precedentes con relación jurídica significativa.", normal_style))
            
            story.append(Spacer(1, 30))
            
            # Firma
            signature_style = ParagraphStyle(
                'Signature',
                parent=styles['Normal'],
                alignment=1,  # Centro
                fontSize=10,
                textColor='gray'
            )
            story.append(Paragraph("Elaborado por EUREKA_TEAM", signature_style))
            
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"Error generando PDF: {e}")
            return False
    
    def analyze_document(self, pdf_path: str) -> Dict:
        """Función principal que analiza un documento completo"""
        try:
            # Extraer texto del PDF
            document_text = self.extract_text_from_pdf(pdf_path)
            
            # Generar resúmenes
            facts_summary = self.generate_facts_summary(document_text)
            legal_summary = self.generate_legal_summary(document_text)
            
            # Los prompts usados (actualizados)
            facts_prompt = """Analiza el documento y genera un resumen CONCISO de hechos y personas en MÁXIMO 3 PÁRRAFOS:
PÁRRAFO 1 - HECHOS CRONOLÓGICOS (máximo 6 líneas): Principales hechos en orden cronológico con fechas.
PÁRRAFO 2 - PERSONAS FISCALIZADAS (máximo 4 líneas): Nombres, cargos y roles de involucrados.
PÁRRAFO 3 - DATOS CLAVE (máximo 4 líneas): Montos económicos, instituciones fiscalizadas y lugares relevantes."""
            
            legal_prompt = """Analiza el documento como abogado experto en materia administrativa de costa rica y extrae los 5 PRINCIPALES ARGUMENTOS JURÍDICOS más relevantes.
Cada argumento debe tener: título y resumen de exactamente 5 líneas explicando normativa citada, 
interpretación legal y conclusión. Enfócate en argumentos jurídicos sólidos con leyes, artículos, 
principios jurídicos y conclusiones legales."""
            
            return {
                'success': True,
                'document_text': document_text,
                'facts_summary': facts_summary,
                'legal_summary': legal_summary,
                'facts_prompt': facts_prompt,
                'legal_prompt': legal_prompt
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }