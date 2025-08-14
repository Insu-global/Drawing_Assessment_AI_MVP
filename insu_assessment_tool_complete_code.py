"""
INSU Architectural Drawing Assessment Tool - Complete Code
==========================================================

This file contains all the code from the INSU Assessment Tool project
combined into a single file for easy sharing and reference.

Project Overview:
- FastAPI-based system for processing architectural drawings
- Supports PDF, DWG, and image formats
- Assesses drawings against Hong Kong and Australian building codes
- Uses AI (GPT-4o) for intelligent compliance checking
- Generates structured compliance reports

Setup Instructions:
1. Install dependencies: pip install -r requirements.txt
2. Set up environment variables in .env file
3. Run: python main.py

Dependencies (requirements.txt):
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
PyMuPDF==1.23.8
Pillow==10.1.0
opencv-python==4.8.1.78
pytesseract==0.3.10
numpy==1.24.3
pandas==2.0.3
langchain==0.0.335
langchain-openai==0.0.2
openai==1.3.7
azure-openai==1.3.0
chromadb==0.4.18
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
alembic==1.12.1
pydantic==2.5.0
python-dotenv==1.0.0
google-api-python-client==2.108.0
google-auth-httplib2==0.1.1
google-auth-oauthlib==1.1.0
reportlab==4.0.7
matplotlib==3.7.2
seaborn==0.12.2
durable-rules==2.0.28
"""

# =============================================================================
# COMBINED SOURCE CODE
# =============================================================================


# =============================================================================
# MAIN.PY
# =============================================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import pytesseract
from PIL import Image
import io
import json
from typing import Dict, List, Any
from pydantic import BaseModel
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Drawing Assessment Tool", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AssessmentResult(BaseModel):
    file_name: str
    assessment_date: datetime
    extracted_text: str
    dimensions: List[str]
    compliance_issues: List[Dict[str, Any]]
    compliance_score: float
    recommendations: List[str]

class DrawingParser:
    def __init__(self):
        # Configure Tesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
        pass
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF using PyPDF2"""
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def extract_text_from_image(self, image_bytes: bytes) -> str:
        """Extract text from image using Tesseract OCR"""
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    
    def extract_dimensions(self, text: str) -> List[str]:
        """Extract dimension measurements from text"""
        import re
        # Pattern to match dimensions like "3.5m", "1200mm", "4'-6"", etc.
        dimension_patterns = [
            r'\d+\.?\d*\s*m\b',  # meters
            r'\d+\.?\d*\s*mm\b',  # millimeters
            r'\d+\.?\d*\s*cm\b',  # centimeters
            r'\d+\'?\-?\d*\"?',   # feet and inches
        ]
        
        dimensions = []
        for pattern in dimension_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dimensions.extend(matches)
        
        return list(set(dimensions))  # Remove duplicates

class ComplianceChecker:
    def __init__(self):
        self.rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules from parsed regulation data"""
        # Try to load from parsed regulations JSON file
        regulations_file = "regulations/parsed_rules.json"
        
        if os.path.exists(regulations_file):
            try:
                with open(regulations_file, 'r') as f:
                    parsed_rules = json.load(f)
                
                # Convert parsed rules to compliance checker format
                rules = self._convert_parsed_rules_to_format(parsed_rules)
                print("Loaded compliance rules from parsed regulations")
                return rules
            except Exception as e:
                print(f"Error loading parsed rules: {e}, using default rules")
        
        # Fallback to default rules if parsed file not available
        return self._get_default_rules()
    
    def _convert_parsed_rules_to_format(self, parsed_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parsed regulation rules to compliance checker format"""
        summary = parsed_rules.get("summary", {})
        
        # Extract values from summary or use defaults
        hk_stair_width = summary.get("staircase_width_mm", {}).get("hong_kong", 1100)
        au_stair_width = summary.get("staircase_width_mm", {}).get("australia", 1000)
        hk_ceiling_height = summary.get("ceiling_height_m", {}).get("hong_kong", 2.5)
        au_ceiling_height = summary.get("ceiling_height_m", {}).get("australia", 2.4)
        hk_exit_width = summary.get("exit_width_mm", {}).get("hong_kong", 850)
        au_exit_width = summary.get("exit_width_mm", {}).get("australia", 850)
        
        return {
            "hong_kong": {
                "building_ordinance": {
                    "staircase_width": {"min": hk_stair_width, "unit": "mm"},
                    "ceiling_height": {"min": hk_ceiling_height, "unit": "m"},
                    "exit_width": {"min": hk_exit_width, "unit": "mm"}
                },
                "fire_safety": {
                    "exit_distance": {"max": summary.get("fire_exit_distance_m", {}).get("hong_kong", 45), "unit": "m"},
                    "corridor_width": {"min": 1050, "unit": "mm"}
                },
                "source_documents": [doc.get("source", "") for doc in parsed_rules.get("hong_kong", [])]
            },
            "australia": {
                "bca": {
                    "staircase_width": {"min": au_stair_width, "unit": "mm"},
                    "ceiling_height": {"min": au_ceiling_height, "unit": "m"}
                },
                "ncc": {
                    "exit_width": {"min": au_exit_width, "unit": "mm"}
                },
                "source_documents": [doc.get("source", "") for doc in parsed_rules.get("australia", [])]
            }
        }
    
    def _get_default_rules(self) -> Dict[str, Any]:
        """Default compliance rules if no parsed data available"""
        return {
            "hong_kong": {
                "building_ordinance": {
                    "staircase_width": {"min": 1100, "unit": "mm"},
                    "ceiling_height": {"min": 2.5, "unit": "m"},
                    "exit_width": {"min": 850, "unit": "mm"}
                },
                "fire_safety": {
                    "exit_distance": {"max": 45, "unit": "m"},
                    "corridor_width": {"min": 1050, "unit": "mm"}
                }
            },
            "australia": {
                "bca": {
                    "staircase_width": {"min": 1000, "unit": "mm"},
                    "ceiling_height": {"min": 2.4, "unit": "m"}
                },
                "ncc": {
                    "exit_width": {"min": 850, "unit": "mm"}
                }
            }
        }
    
    def check_compliance(self, text: str, dimensions: List[str]) -> Dict[str, Any]:
        """Check compliance against statutory codes"""
        issues = []
        score = 100.0
        
        # Simple rule-based checking (to be enhanced with AI)
        if "staircase" in text.lower() or "stair" in text.lower():
            # Check staircase width compliance
            stair_widths = [d for d in dimensions if 'mm' in d and any(keyword in text.lower() for keyword in ['stair', 'step'])]
            if stair_widths:
                for width_str in stair_widths:
                    width = float(width_str.replace('mm', '').strip())
                    if width < 1100:  # Hong Kong minimum
                        issues.append({
                            "type": "staircase_width",
                            "severity": "high",
                            "description": f"Staircase width {width}mm is below Hong Kong minimum of 1100mm",
                            "code_reference": "Building Ordinance"
                        })
                        score -= 20
        
        if "ceiling" in text.lower():
            ceiling_heights = [d for d in dimensions if 'm' in d and 'ceiling' in text.lower()]
            for height_str in ceiling_heights:
                height = float(height_str.replace('m', '').strip())
                if height < 2.5:
                    issues.append({
                        "type": "ceiling_height",
                        "severity": "medium",
                        "description": f"Ceiling height {height}m is below minimum of 2.5m",
                        "code_reference": "Building Ordinance"
                    })
                    score -= 15
        
        recommendations = []
        if issues:
            recommendations.append("Review dimensions against statutory requirements")
            recommendations.append("Consult with structural engineer for compliance verification")
        
        return {
            "issues": issues,
            "score": max(0, score),
            "recommendations": recommendations
        }

@app.post("/upload", response_model=AssessmentResult)
async def upload_and_assess(file: UploadFile = File(...)):
    """Upload and assess architectural drawing"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file type
    allowed_types = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Initialize components
        parser = DrawingParser()
        checker = ComplianceChecker()
        
        # Extract text based on file type
        if file_ext == ".pdf":
            extracted_text = parser.extract_text_from_pdf(file_content)
        else:
            extracted_text = parser.extract_text_from_image(file_content)
        
        # Extract dimensions
        dimensions = parser.extract_dimensions(extracted_text)
        
        # Check compliance
        compliance_result = checker.check_compliance(extracted_text, dimensions)
        
        # Create assessment result
        result = AssessmentResult(
            file_name=file.filename,
            assessment_date=datetime.now(),
            extracted_text=extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,
            dimensions=dimensions,
            compliance_issues=compliance_result["issues"],
            compliance_score=compliance_result["score"],
            recommendations=compliance_result["recommendations"]
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Drawing Assessment Tool API", "status": "active"}

@app.get("/rules")
async def get_rules():
    """Get available compliance rules"""
    checker = ComplianceChecker()
    return checker.rules

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# =============================================================================
# AI_AGENT.PY
# =============================================================================

from typing import Dict, Any, List, Optional
import openai
from dotenv import load_dotenv
import os
import json

load_dotenv()

class AIComplianceAgent:
    """
    AI Agent for advanced compliance assessment using GPT-4o
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.model = "gpt-4o-mini"  # Will be upgraded to gpt-4o when available
        
    def assess_compliance_with_ai(self, 
                                extracted_text: str, 
                                dimensions: List[str],
                                drawing_type: str = "architectural") -> Dict[str, Any]:
        """
        Use AI to perform advanced compliance assessment
        
        Args:
            extracted_text: Text extracted from the drawing
            dimensions: List of dimensions found
            drawing_type: Type of drawing (architectural, structural, etc.)
            
        Returns:
            Dictionary with AI assessment results
        """
        
        prompt = self._create_compliance_prompt(extracted_text, dimensions, drawing_type)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert building code compliance assessor specializing in Hong Kong Building Ordinance, PNAP, Fire Safety Codes, Australian BCA, and NCC. Analyze architectural drawings for compliance issues and provide detailed assessments."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            ai_assessment = response.choices[0].message.content
            
            # Parse the AI response into structured format
            structured_result = self._parse_ai_response(ai_assessment)
            
            return {
                "ai_assessment": ai_assessment,
                "structured_issues": structured_result.get("issues", []),
                "compliance_score": structured_result.get("score", 0),
                "recommendations": structured_result.get("recommendations", []),
                "code_references": structured_result.get("code_references", []),
                "model_used": self.model
            }
            
        except Exception as e:
            return {
                "error": f"AI assessment failed: {str(e)}",
                "ai_assessment": "",
                "structured_issues": [],
                "compliance_score": 0,
                "recommendations": ["Manual review required due to AI assessment failure"],
                "code_references": [],
                "model_used": self.model
            }
    
    def _create_compliance_prompt(self, text: str, dimensions: List[str], drawing_type: str) -> str:
        """Create a detailed prompt for AI compliance assessment"""
        
        prompt = f"""
        Please analyze the following architectural drawing content for compliance with Hong Kong and Australian building codes:

        **Drawing Type:** {drawing_type}

        **Extracted Text:**
        {text[:2000]}  # Limit text to avoid token limits

        **Extracted Dimensions:**
        {', '.join(dimensions[:50])}  # Limit dimensions

        **Assessment Requirements:**
        
        1. **Hong Kong Codes:**
           - Building Ordinance (BO)
           - Practice Notes for Authorized Persons (PNAP)
           - Fire Safety Code

        2. **Australian Codes:**
           - Building Code of Australia (BCA)
           - National Construction Code (NCC)

        **Specific Areas to Check:**
        - Staircase widths (min 1100mm HK, 1000mm AU)
        - Ceiling heights (min 2.5m HK, 2.4m AU)
        - Exit widths and egress paths
        - Fire safety requirements
        - Accessibility compliance
        - Structural safety indicators

        **Required Output Format:**
        Please provide your assessment in the following JSON-like structure:

        ```json
        {{
            "overall_compliance_score": <0-100>,
            "issues": [
                {{
                    "type": "issue_category",
                    "severity": "high|medium|low",
                    "description": "detailed description",
                    "code_reference": "specific code section",
                    "location": "where in drawing if identifiable",
                    "recommendation": "how to fix"
                }}
            ],
            "recommendations": [
                "general recommendation 1",
                "general recommendation 2"
            ],
            "code_references": [
                "BO Section X.X",
                "BCA Section Y.Y"
            ],
            "notes": "any additional observations"
        }}
        ```

        Focus on identifying actual compliance issues based on the extracted content and dimensions.
        """
        
        return prompt
    
    def _parse_ai_response(self, ai_response: str) -> Dict[str, Any]:
        """
        Parse AI response and extract structured information
        """
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', ai_response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                parsed = json.loads(json_str)
                return parsed
            else:
                # Fallback: try to parse the entire response as JSON
                return json.loads(ai_response)
                
        except json.JSONDecodeError:
            # Fallback: extract information using regex patterns
            return self._extract_info_with_regex(ai_response)
    
    def _extract_info_with_regex(self, text: str) -> Dict[str, Any]:
        """
        Extract information using regex patterns when JSON parsing fails
        """
        import re
        
        # Extract compliance score
        score_match = re.search(r'(?:score|compliance).*?(\d+)', text, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 50
        
        # Extract issues (simple pattern matching)
        issue_patterns = [
            r'(?:issue|problem|violation).*?:(.*?)(?:\n|$)',
            r'(?:non-compliant|fails to meet).*?:(.*?)(?:\n|$)'
        ]
        
        issues = []
        for pattern in issue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                issues.append({
                    "type": "general",
                    "severity": "medium",
                    "description": match.strip(),
                    "code_reference": "Unknown",
                    "recommendation": "Review and correct"
                })
        
        # Extract recommendations
        rec_pattern = r'(?:recommend|suggest).*?:(.*?)(?:\n|$)'
        recommendations = [match.strip() for match in re.findall(rec_pattern, text, re.IGNORECASE)]
        
        return {
            "score": score,
            "issues": issues[:5],  # Limit to 5 issues
            "recommendations": recommendations[:3],  # Limit to 3 recommendations
            "code_references": ["AI-identified"]
        }
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the AI model being used"""
        return {
            "current_model": self.model,
            "upgrade_path": "gpt-4o (when available)",
            "capabilities": "Advanced reasoning, code compliance assessment, architectural analysis",
            "note": "Future integration with GPT-4o will provide enhanced spatial reasoning and drawing analysis"
        }


# =============================================================================
# DWG_PARSER.PY
# =============================================================================

import subprocess
import os
from typing import Optional, Dict, Any
import tempfile

class DWGParser:
    """
    DWG file parser using external tools.
    Note: This requires external dependencies like LibreCAD or AutoCAD
    """
    
    def __init__(self):
        self.supported_converters = {
            "libre_cad": self._check_libre_cad(),
            "dxf_converter": self._check_dxf_converter()
        }
    
    def _check_libre_cad(self) -> bool:
        """Check if LibreCAD is available"""
        try:
            result = subprocess.run(["librecad", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _check_dxf_converter(self) -> bool:
        """Check if DXF converter tools are available"""
        try:
            # Check for common DXF conversion tools
            for tool in ["dxf2pdf", "dwg2dxf", "qcad"]:
                result = subprocess.run([tool, "--help"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return True
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def can_parse_dwg(self) -> bool:
        """Check if DWG parsing is supported"""
        return any(self.supported_converters.values())
    
    def parse_dwg_file(self, dwg_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Parse DWG file and extract text/metadata
        
        Args:
            dwg_content: Raw DWG file content
            filename: Original filename
            
        Returns:
            Dictionary with extracted information
        """
        if not self.can_parse_dwg():
            return {
                "error": "No DWG parsing tools available",
                "text": "",
                "metadata": {},
                "conversion_method": None
            }
        
        # Create temporary file for DWG content
        with tempfile.NamedTemporaryFile(suffix=".dwg", delete=False) as temp_dwg:
            temp_dwg.write(dwg_content)
            temp_dwg_path = temp_dwg.name
        
        try:
            # Try different conversion methods
            if self.supported_converters["libre_cad"]:
                return self._parse_with_librecad(temp_dwg_path, filename)
            elif self.supported_converters["dxf_converter"]:
                return self._parse_with_dxf_converter(temp_dwg_path, filename)
            else:
                return {
                    "error": "No suitable DWG parser found",
                    "text": "",
                    "metadata": {},
                    "conversion_method": None
                }
        finally:
            # Clean up temporary file
            if os.path.exists(temp_dwg_path):
                os.unlink(temp_dwg_path)
    
    def _parse_with_librecad(self, dwg_path: str, filename: str) -> Dict[str, Any]:
        """Parse DWG using LibreCAD"""
        try:
            # Convert DWG to DXF using LibreCAD command line
            with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as temp_dxf:
                temp_dxf_path = temp_dxf.name
            
            # LibreCAD batch conversion command
            cmd = ["librecad", "-b", dwg_path, temp_dxf_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return {
                    "error": f"LibreCAD conversion failed: {result.stderr}",
                    "text": "",
                    "metadata": {},
                    "conversion_method": "librecad"
                }
            
            # Parse the resulting DXF file
            extracted_text = self._extract_text_from_dxf(temp_dxf_path)
            
            # Clean up
            if os.path.exists(temp_dxf_path):
                os.unlink(temp_dxf_path)
            
            return {
                "text": extracted_text,
                "metadata": {
                    "original_format": "dwg",
                    "converted_to": "dxf",
                    "filename": filename
                },
                "conversion_method": "librecad",
                "error": None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "error": "LibreCAD conversion timeout",
                "text": "",
                "metadata": {},
                "conversion_method": "librecad"
            }
        except Exception as e:
            return {
                "error": f"LibreCAD parsing error: {str(e)}",
                "text": "",
                "metadata": {},
                "conversion_method": "librecad"
            }
    
    def _parse_with_dxf_converter(self, dwg_path: str, filename: str) -> Dict[str, Any]:
        """Parse DWG using DXF converter tools"""
        # This is a placeholder for other DXF conversion tools
        # Implementation would depend on the specific tool available
        return {
            "error": "DXF converter not implemented yet",
            "text": "",
            "metadata": {},
            "conversion_method": "dxf_converter"
        }
    
    def _extract_text_from_dxf(self, dxf_path: str) -> str:
        """Extract text entities from DXF file"""
        try:
            with open(dxf_path, 'r', encoding='utf-8', errors='ignore') as f:
                dxf_content = f.read()
            
            # Simple text extraction from DXF
            # Look for TEXT and MTEXT entities
            text_entities = []
            lines = dxf_content.split('\n')
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for TEXT entities
                if line == "TEXT" or line == "MTEXT":
                    # Skip to find the actual text value (usually after code 1)
                    for j in range(i + 1, min(i + 20, len(lines))):
                        if lines[j].strip() == "1" and j + 1 < len(lines):
                            text_value = lines[j + 1].strip()
                            if text_value and text_value not in ["TEXT", "MTEXT", "1"]:
                                text_entities.append(text_value)
                            break
                i += 1
            
            return "\n".join(text_entities)
            
        except Exception as e:
            return f"Error reading DXF file: {str(e)}"
    
    def get_installation_instructions(self) -> Dict[str, str]:
        """Get installation instructions for DWG parsing tools"""
        return {
            "librecad": "Install LibreCAD: brew install librecad (macOS) or apt-get install librecad (Ubuntu)",
            "qcad": "Install QCAD: https://qcad.org/en/download",
            "autocad": "AutoCAD command line tools (commercial license required)",
            "note": "DWG parsing requires external CAD software due to proprietary format restrictions"
        }


# =============================================================================
# REGULATION_PARSER.PY
# =============================================================================

import PyPDF2
import re
import json
from typing import Dict, List, Any, Optional
import os
from pathlib import Path

class BuildingRegulationParser:
    """
    Parser for building regulation documents to extract compliance rules
    """
    
    def __init__(self, regulations_folder: str = "regulations/"):
        self.regulations_folder = regulations_folder
        self.extracted_rules = {}
        
    def parse_pdf_regulations(self, pdf_path: str) -> Dict[str, Any]:
        """Parse building regulations from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            # Extract rules based on document type
            file_name = os.path.basename(pdf_path).lower()
            
            if 'hong_kong' in file_name or 'building_ordinance' in file_name:
                return self._parse_hong_kong_regulations(text, pdf_path)
            elif 'australia' in file_name or 'bca' in file_name or 'ncc' in file_name:
                return self._parse_australia_regulations(text, pdf_path)
            else:
                return self._parse_generic_regulations(text, pdf_path)
                
        except Exception as e:
            print(f"Error parsing {pdf_path}: {e}")
            return {"error": str(e), "source": pdf_path}
    
    def _parse_hong_kong_regulations(self, text: str, source: str) -> Dict[str, Any]:
        """Parse Hong Kong building regulations"""
        rules = {
            "jurisdiction": "Hong Kong",
            "source": source,
            "staircase_requirements": {},
            "fire_safety_requirements": {},
            "building_height_requirements": {},
            "accessibility_requirements": {}
        }
        
        # Extract staircase width requirements
        stair_patterns = [
            r'staircase.*?width.*?(\d+\.?\d*)\s*(mm|m)',
            r'stair.*?minimum.*?(\d+\.?\d*)\s*(mm|m)',
            r'step.*?width.*?(\d+\.?\d*)\s*(mm|m)'
        ]
        
        for pattern in stair_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for value, unit in matches:
                    rules["staircase_requirements"][f"width_{unit}"] = float(value)
        
        # Extract ceiling height requirements
        ceiling_patterns = [
            r'ceiling.*?height.*?(\d+\.?\d*)\s*(m|mm)',
            r'room.*?height.*?(\d+\.?\d*)\s*(m|mm)',
            r'minimum.*?height.*?(\d+\.?\d*)\s*(m|mm)'
        ]
        
        for pattern in ceiling_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for value, unit in matches:
                    rules["building_height_requirements"][f"ceiling_{unit}"] = float(value)
        
        # Extract fire safety requirements
        fire_patterns = [
            r'fire.*?exit.*?(\d+\.?\d*)\s*(m|mm)',
            r'egress.*?width.*?(\d+\.?\d*)\s*(m|mm)',
            r'escape.*?route.*?(\d+\.?\d*)\s*(m|mm)'
        ]
        
        for pattern in fire_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for value, unit in matches:
                    rules["fire_safety_requirements"][f"exit_{unit}"] = float(value)
        
        return rules
    
    def _parse_australia_regulations(self, text: str, source: str) -> Dict[str, Any]:
        """Parse Australian building regulations (BCA/NCC)"""
        rules = {
            "jurisdiction": "Australia",
            "source": source,
            "staircase_requirements": {},
            "fire_safety_requirements": {},
            "building_height_requirements": {},
            "accessibility_requirements": {}
        }
        
        # Similar pattern extraction for Australian codes
        # Australian standards typically have different minimum requirements
        
        # BCA staircase requirements
        stair_patterns = [
            r'stairway.*?width.*?(\d+\.?\d*)\s*(mm|m)',
            r'stair.*?minimum.*?(\d+\.?\d*)\s*(mm|m)',
            r'effective.*?width.*?(\d+\.?\d*)\s*(mm|m)'
        ]
        
        for pattern in stair_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for value, unit in matches:
                    rules["staircase_requirements"][f"width_{unit}"] = float(value)
        
        return rules
    
    def _parse_generic_regulations(self, text: str, source: str) -> Dict[str, Any]:
        """Parse generic building regulations"""
        rules = {
            "jurisdiction": "Generic",
            "source": source,
            "extracted_dimensions": [],
            "requirements": {}
        }
        
        # Extract all dimensions mentioned in the document
        dimension_patterns = [
            r'(\d+\.?\d*)\s*(mm|m|cm|inches?|feet?|ft)',
            r'minimum.*?(\d+\.?\d*)\s*(mm|m|cm)',
            r'maximum.*?(\d+\.?\d*)\s*(mm|m|cm)'
        ]
        
        for pattern in dimension_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            rules["extracted_dimensions"].extend([f"{value}{unit}" for value, unit in matches])
        
        return rules
    
    def parse_all_regulations(self) -> Dict[str, Any]:
        """Parse all regulation files in the regulations folder"""
        if not os.path.exists(self.regulations_folder):
            return {"error": f"Regulations folder {self.regulations_folder} not found"}
        
        all_rules = {
            "hong_kong": [],
            "australia": [],
            "generic": [],
            "summary": {}
        }
        
        # Process all PDF files in the regulations folder
        for file_path in Path(self.regulations_folder).glob("*.pdf"):
            print(f"Parsing: {file_path}")
            rules = self.parse_pdf_regulations(str(file_path))
            
            if "error" not in rules:
                jurisdiction = rules.get("jurisdiction", "generic").lower()
                if jurisdiction == "hong kong":
                    all_rules["hong_kong"].append(rules)
                elif jurisdiction == "australia":
                    all_rules["australia"].append(rules)
                else:
                    all_rules["generic"].append(rules)
        
        # Create summary of requirements
        all_rules["summary"] = self._create_requirements_summary(all_rules)
        
        # Save parsed rules
        self._save_parsed_rules(all_rules)
        
        return all_rules
    
    def _create_requirements_summary(self, all_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of all requirements for quick compliance checking"""
        summary = {
            "staircase_width_mm": {"hong_kong": 1100, "australia": 1000},
            "ceiling_height_m": {"hong_kong": 2.5, "australia": 2.4},
            "exit_width_mm": {"hong_kong": 850, "australia": 850},
            "fire_exit_distance_m": {"hong_kong": 45, "australia": 60}
        }
        
        # Update with parsed values if found
        for jurisdiction in ["hong_kong", "australia"]:
            for rules_set in all_rules.get(jurisdiction, []):
                stair_reqs = rules_set.get("staircase_requirements", {})
                if "width_mm" in stair_reqs:
                    summary["staircase_width_mm"][jurisdiction] = stair_reqs["width_mm"]
                
                height_reqs = rules_set.get("building_height_requirements", {})
                if "ceiling_m" in height_reqs:
                    summary["ceiling_height_m"][jurisdiction] = height_reqs["ceiling_m"]
        
        return summary
    
    def _save_parsed_rules(self, rules: Dict[str, Any]):
        """Save parsed rules to JSON file"""
        output_file = os.path.join(self.regulations_folder, "parsed_rules.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(rules, f, indent=2, default=str)
            print(f"Parsed rules saved to: {output_file}")
        except Exception as e:
            print(f"Error saving parsed rules: {e}")
    
    def get_compliance_rules_for_checker(self) -> Dict[str, Any]:
        """Get rules in format suitable for compliance checker"""
        parsed_file = os.path.join(self.regulations_folder, "parsed_rules.json")
        
        if os.path.exists(parsed_file):
            with open(parsed_file, 'r') as f:
                rules = json.load(f)
            return rules.get("summary", {})
        else:
            # Return default rules if no parsed file exists
            return {
                "staircase_width_mm": {"hong_kong": 1100, "australia": 1000},
                "ceiling_height_m": {"hong_kong": 2.5, "australia": 2.4},
                "exit_width_mm": {"hong_kong": 850, "australia": 850}
            }

# Usage example
if __name__ == "__main__":
    parser = BuildingRegulationParser()
    
    # Parse all regulation documents
    results = parser.parse_all_regulations()
    
    print("Parsing Results:")
    print(f"Hong Kong regulations: {len(results['hong_kong'])} files")
    print(f"Australia regulations: {len(results['australia'])} files")
    print(f"Generic regulations: {len(results['generic'])} files")
    
    # Get summary for compliance checking
    compliance_rules = parser.get_compliance_rules_for_checker()
    print("\nCompliance Rules Summary:")
    print(json.dumps(compliance_rules, indent=2))


# =============================================================================
# REPORT_GENERATOR.PY
# =============================================================================

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import json
import io
from typing import Dict, Any, List

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        
    def generate_json_report(self, assessment_data: Dict[str, Any]) -> str:
        """Generate JSON format report"""
        report = {
            "assessment_metadata": {
                "file_name": assessment_data.get("file_name"),
                "assessment_date": assessment_data.get("assessment_date").isoformat() if assessment_data.get("assessment_date") else None,
                "tool_version": "1.0.0"
            },
            "extracted_content": {
                "text_length": len(assessment_data.get("extracted_text", "")),
                "dimensions_found": len(assessment_data.get("dimensions", [])),
                "dimensions": assessment_data.get("dimensions", [])
            },
            "compliance_assessment": {
                "overall_score": assessment_data.get("compliance_score", 0),
                "total_issues": len(assessment_data.get("compliance_issues", [])),
                "issues_by_severity": self._categorize_issues_by_severity(assessment_data.get("compliance_issues", [])),
                "detailed_issues": assessment_data.get("compliance_issues", [])
            },
            "recommendations": assessment_data.get("recommendations", []),
            "code_references": self._extract_code_references(assessment_data.get("compliance_issues", []))
        }
        return json.dumps(report, indent=2, default=str)
    
    def generate_pdf_report(self, assessment_data: Dict[str, Any]) -> bytes:
        """Generate PDF format report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
        story = []
        
        # Title
        title = Paragraph("Architectural Drawing Compliance Assessment Report", self.title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Assessment metadata
        metadata_data = [
            ["File Name:", assessment_data.get("file_name", "N/A")],
            ["Assessment Date:", assessment_data.get("assessment_date", "N/A")],
            ["Overall Compliance Score:", f"{assessment_data.get('compliance_score', 0):.1f}%"]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Compliance Issues
        story.append(Paragraph("Compliance Issues", self.styles['Heading2']))
        story.append(Spacer(1, 10))
        
        issues = assessment_data.get("compliance_issues", [])
        if issues:
            issue_data = [["Type", "Severity", "Description", "Code Reference"]]
            for issue in issues:
                issue_data.append([
                    issue.get("type", ""),
                    issue.get("severity", ""),
                    issue.get("description", ""),
                    issue.get("code_reference", "")
                ])
            
            issue_table = Table(issue_data, colWidths=[1.5*inch, 1*inch, 3*inch, 1.5*inch])
            issue_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(issue_table)
        else:
            story.append(Paragraph("No compliance issues found.", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Dimensions Found
        story.append(Paragraph("Extracted Dimensions", self.styles['Heading2']))
        story.append(Spacer(1, 10))
        
        dimensions = assessment_data.get("dimensions", [])
        if dimensions:
            dim_text = ", ".join(dimensions[:20])  # Limit to first 20 dimensions
            if len(dimensions) > 20:
                dim_text += f" ... and {len(dimensions) - 20} more"
            story.append(Paragraph(dim_text, self.styles['Normal']))
        else:
            story.append(Paragraph("No dimensions found in the drawing.", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.styles['Heading2']))
        story.append(Spacer(1, 10))
        
        recommendations = assessment_data.get("recommendations", [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", self.styles['Normal']))
                story.append(Spacer(1, 5))
        else:
            story.append(Paragraph("No specific recommendations at this time.", self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _categorize_issues_by_severity(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize issues by severity level"""
        categories = {"high": 0, "medium": 0, "low": 0}
        for issue in issues:
            severity = issue.get("severity", "low")
            if severity in categories:
                categories[severity] += 1
        return categories
    
    def _extract_code_references(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Extract unique code references from issues"""
        references = set()
        for issue in issues:
            ref = issue.get("code_reference")
            if ref:
                references.add(ref)
        return list(references)


# =============================================================================
# SRC/CORE/PARSERS/PDF_PARSER.PY
# =============================================================================

import fitz  # PyMuPDF
import re
from typing import Dict, List, Any, Optional
import requests
from urllib.parse import urlparse, parse_qs
import os

class FireSafetyCodeParser:
    """Parser for Fire Safety Code PDF from Google Drive"""
    
    def __init__(self, pdf_path_or_url: str):
        self.pdf_path_or_url = pdf_path_or_url
        self.doc = None
        self.text_content = ""
        self.tables = []
        self.clauses = {}
        
    def load_pdf(self) -> bool:
        """Load PDF from local path or Google Drive URL"""
        try:
            if self.pdf_path_or_url.startswith('http'):
                # Handle Google Drive URL
                self.doc = self._load_from_google_drive(self.pdf_path_or_url)
            else:
                # Handle local file
                self.doc = fitz.open(self.pdf_path_or_url)
            
            if self.doc:
                self._extract_text()
                return True
            return False
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return False
    
    def _load_from_google_drive(self, url: str) -> Optional[fitz.Document]:
        """Load PDF from Google Drive URL"""
        try:
            # Extract file ID from Google Drive URL
            file_id = self._extract_file_id(url)
            if not file_id:
                print("Could not extract file ID from Google Drive URL")
                return None
            
            # Create direct download URL
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            # Download the file
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            # Save temporarily and open with PyMuPDF
            temp_path = f"temp_fire_safety_code_{file_id}.pdf"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            doc = fitz.open(temp_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
            return doc
            
        except Exception as e:
            print(f"Error loading from Google Drive: {e}")
            return None
    
    def _extract_file_id(self, url: str) -> Optional[str]:
        """Extract file ID from Google Drive URL"""
        try:
            parsed = urlparse(url)
            if 'drive.google.com' in parsed.netloc:
                if '/file/d/' in url:
                    # Format: https://drive.google.com/file/d/FILE_ID/view
                    file_id = url.split('/file/d/')[1].split('/')[0]
                    return file_id
                elif 'id=' in parsed.query:
                    # Format: https://drive.google.com/open?id=FILE_ID
                    query_params = parse_qs(parsed.query)
                    return query_params.get('id', [None])[0]
        except Exception as e:
            print(f"Error extracting file ID: {e}")
        return None
    
    def _extract_text(self):
        """Extract all text from PDF"""
        if not self.doc:
            return
        
        self.text_content = ""
        for page in self.doc:
            self.text_content += page.get_text()
    
    def extract_staircase_requirements(self) -> Dict[str, Any]:
        """Extract staircase discharge requirements from Fire Safety Code"""
        requirements = {
            "tables": [],
            "clauses": [],
            "minimum_widths": {},
            "exceptions": []
        }
        
        if not self.text_content:
            return requirements
        
        # Look for staircase-related content
        staircase_patterns = [
            r"staircase\s+discharge\s+width",
            r"stair\s+width",
            r"stairway\s+width",
            r"exit\s+stair",
            r"fire\s+escape\s+stair"
        ]
        
        # Extract relevant sections
        lines = self.text_content.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check for staircase-related content
            if any(re.search(pattern, line_lower) for pattern in staircase_patterns):
                # Get context (previous and next lines)
                context_start = max(0, i-5)
                context_end = min(len(lines), i+6)
                context = lines[context_start:context_end]
                
                requirements["clauses"].append({
                    "line_number": i+1,
                    "content": line.strip(),
                    "context": context
                })
            
            # Look for tables with width requirements
            if re.search(r"table\s+\d+", line_lower) and any(pattern in line_lower for pattern in staircase_patterns):
                requirements["tables"].append({
                    "line_number": i+1,
                    "content": line.strip()
                })
            
            # Extract minimum width values
            width_matches = re.findall(r"(\d+)\s*(?:mm|millimeters?)\s*(?:minimum|min|width)", line_lower)
            for match in width_matches:
                requirements["minimum_widths"][f"line_{i+1}"] = {
                    "value": int(match),
                    "context": line.strip()
                }
        
        return requirements
    
    def search_specific_clause(self, clause_number: str) -> List[Dict[str, Any]]:
        """Search for specific clause number"""
        results = []
        if not self.text_content:
            return results
        
        lines = self.text_content.split('\n')
        for i, line in enumerate(lines):
            if clause_number.lower() in line.lower():
                # Get context
                context_start = max(0, i-3)
                context_end = min(len(lines), i+4)
                context = lines[context_start:context_end]
                
                results.append({
                    "line_number": i+1,
                    "content": line.strip(),
                    "context": context
                })
        
        return results
    
    def search_table(self, table_number: str) -> List[Dict[str, Any]]:
        """Search for specific table"""
        results = []
        if not self.text_content:
            return results
        
        lines = self.text_content.split('\n')
        in_table = False
        table_content = []
        
        for i, line in enumerate(lines):
            if re.search(f"table\s+{table_number}", line.lower()):
                in_table = True
                table_content = [line.strip()]
            elif in_table:
                if re.search(r"table\s+\d+", line.lower()) or re.search(r"section\s+\d+", line.lower()):
                    # End of table
                    break
                else:
                    table_content.append(line.strip())
        
        if table_content:
            results.append({
                "table_number": table_number,
                "content": table_content
            })
        
        return results
    
    def get_compliance_requirements(self, building_type: str = "general", 
                                  occupant_load: int = None) -> Dict[str, Any]:
        """Get specific compliance requirements based on building characteristics"""
        requirements = {
            "minimum_width": None,
            "clause_reference": "",
            "table_reference": "",
            "exceptions": [],
            "notes": []
        }
        
        # Search for relevant clauses
        clause_results = self.search_specific_clause("305")  # Staircase clause
        if clause_results:
            requirements["clause_reference"] = f"Section 305 - Found in {len(clause_results)} locations"
            requirements["notes"].append("Section 305 contains staircase requirements")
        
        # Search for relevant tables
        table_results = self.search_table("3")  # Common table for staircase requirements
        if table_results:
            requirements["table_reference"] = f"Table 3 - Found staircase width requirements"
            requirements["notes"].append("Table 3 contains width specifications")
        
        # Extract minimum width from content
        staircase_req = self.extract_staircase_requirements()
        if staircase_req["minimum_widths"]:
            # Get the highest minimum width found
            max_width = max([req["value"] for req in staircase_req["minimum_widths"].values()])
            requirements["minimum_width"] = max_width
            requirements["notes"].append(f"Minimum width requirement: {max_width}mm")
        
        return requirements
    
    def close(self):
        """Close the PDF document"""
        if self.doc:
            self.doc.close()

def extract_fire_safety_requirements(pdf_path_or_url: str, 
                                   building_type: str = "general",
                                   occupant_load: int = None) -> Dict[str, Any]:
    """Extract Fire Safety Code requirements for staircase discharge"""
    
    parser = FireSafetyCodeParser(pdf_path_or_url)
    
    if not parser.load_pdf():
        return {"error": "Could not load PDF"}
    
    try:
        # Get general staircase requirements
        staircase_req = parser.extract_staircase_requirements()
        
        # Get specific compliance requirements
        compliance_req = parser.get_compliance_requirements(building_type, occupant_load)
        
        return {
            "staircase_requirements": staircase_req,
            "compliance_requirements": compliance_req,
            "source": pdf_path_or_url
        }
    
    finally:
        parser.close()

# Example usage functions
def test_pdf_parser():
    """Test the PDF parser with a sample"""
    print("=== PDF Parser Test ===")
    print("To use with Google Drive PDF:")
    print("1. Share your Fire Safety Code PDF on Google Drive")
    print("2. Get the sharing link")
    print("3. Use the link in the parser")
    print()
    print("Example:")
    print("url = 'https://drive.google.com/file/d/YOUR_FILE_ID/view'")
    print("requirements = extract_fire_safety_requirements(url)")

if __name__ == "__main__":
    test_pdf_parser() 


# =============================================================================
# SRC/CORE/PARSERS/AREA_DETECTOR.PY
# =============================================================================

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import pytesseract
from PIL import Image
import io
import math
from dataclasses import dataclass
import logging

@dataclass
class DetectedArea:
    """Represents a detected area in a drawing"""
    polygon: Polygon
    area_pixels: float
    area_real: Optional[float]  # In m² if scale is known
    centroid: Tuple[float, float]
    label: Optional[str]
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    contour_points: List[Tuple[int, int]]

class AdvancedAreaDetector:
    """Advanced area detection for scanned architectural drawings"""
    
    def __init__(self, 
                 min_area_pixels: int = 1000,
                 max_area_pixels: int = 500000,
                 scale_factor: Optional[float] = None,
                 drawing_units_per_pixel: Optional[float] = None):
        """
        Initialize the area detector
        
        Args:
            min_area_pixels: Minimum area in pixels to consider
            max_area_pixels: Maximum area in pixels to consider
            scale_factor: Drawing scale (e.g., 100 for 1:100)
            drawing_units_per_pixel: How many drawing units (mm) per pixel
        """
        self.min_area_pixels = min_area_pixels
        self.max_area_pixels = max_area_pixels
        self.scale_factor = scale_factor
        self.drawing_units_per_pixel = drawing_units_per_pixel
        self.logger = logging.getLogger(__name__)
        
    def detect_areas_from_image(self, image_data: bytes) -> List[DetectedArea]:
        """
        Detect enclosed areas from image data
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            List of detected areas
        """
        # Convert bytes to OpenCV image
        image = self._bytes_to_cv2_image(image_data)
        if image is None:
            return []
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Detect contours
        contours = self._detect_contours(processed_image)
        
        # Filter and process contours
        valid_contours = self._filter_contours(contours)
        
        # Extract text labels
        text_blocks = self._extract_text_with_positions(image)
        
        # Create DetectedArea objects
        detected_areas = []
        for contour in valid_contours:
            area = self._create_detected_area(contour, text_blocks, image.shape)
            if area:
                detected_areas.append(area)
        
        return detected_areas
    
    def _bytes_to_cv2_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """Convert image bytes to OpenCV format"""
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format (BGR)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return cv_image
        except Exception as e:
            self.logger.error(f"Error converting image: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better contour detection
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding for better line detection
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        
        # Close gaps in lines
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opened
    
    def _detect_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Detect contours in the binary image
        
        Args:
            binary_image: Preprocessed binary image
            
        Returns:
            List of contours
        """
        # Find contours
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
    
    def _filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter contours based on area and shape criteria
        
        Args:
            contours: Raw contours
            
        Returns:
            Filtered valid contours
        """
        valid_contours = []
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area_pixels or area > self.max_area_pixels:
                continue
            
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Calculate circularity (4π*area/perimeter²)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Filter out very circular shapes (likely not rooms)
            if circularity > 0.85:
                continue
            
            # Check if contour is closed and has reasonable complexity
            if len(contour) < 4:  # Need at least 4 points for a room
                continue
            
            # Approximate contour to reduce noise
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if approximated contour still has reasonable complexity
            if len(approx) >= 3:
                valid_contours.append(approx)
        
        return valid_contours
    
    def _extract_text_with_positions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text with position information using Tesseract
        
        Args:
            image: Input image
            
        Returns:
            List of text blocks with positions
        """
        try:
            # Convert to PIL Image for Tesseract
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Use Tesseract to get text with bounding boxes
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            text_blocks = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and len(text) > 1:  # Filter out single characters
                    confidence = float(data['conf'][i])
                    if confidence > 30:  # Only consider high-confidence text
                        text_blocks.append({
                            'text': text,
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i],
                            'confidence': confidence,
                            'center': (
                                data['left'][i] + data['width'][i] // 2,
                                data['top'][i] + data['height'][i] // 2
                            )
                        })
            
            return text_blocks
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return []
    
    def _create_detected_area(self, 
                            contour: np.ndarray, 
                            text_blocks: List[Dict[str, Any]], 
                            image_shape: Tuple[int, int, int]) -> Optional[DetectedArea]:
        """
        Create a DetectedArea object from a contour
        
        Args:
            contour: OpenCV contour
            text_blocks: List of text blocks with positions
            image_shape: Shape of the original image
            
        Returns:
            DetectedArea object or None if invalid
        """
        try:
            # Convert contour to polygon
            points = [(int(point[0][0]), int(point[0][1])) for point in contour]
            if len(points) < 3:
                return None
            
            polygon = Polygon(points)
            if not polygon.is_valid:
                # Try to fix invalid polygon
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    return None
            
            # Calculate area in pixels
            area_pixels = polygon.area
            
            # Calculate real area if scale is known
            area_real = None
            if self.scale_factor and self.drawing_units_per_pixel:
                # Convert pixels to drawing units (mm), then to m²
                area_mm2 = area_pixels * (self.drawing_units_per_pixel ** 2)
                area_real = area_mm2 * (self.scale_factor ** 2) / (1000 * 1000)
            
            # Get centroid
            centroid = polygon.centroid
            centroid_coords = (centroid.x, centroid.y)
            
            # Find nearest text label
            label = self._find_nearest_label(centroid_coords, text_blocks)
            
            # Calculate confidence based on various factors
            confidence = self._calculate_confidence(polygon, contour, area_pixels)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            return DetectedArea(
                polygon=polygon,
                area_pixels=area_pixels,
                area_real=area_real,
                centroid=centroid_coords,
                label=label,
                confidence=confidence,
                bounding_box=(x, y, w, h),
                contour_points=points
            )
            
        except Exception as e:
            self.logger.error(f"Error creating detected area: {e}")
            return None
    
    def _find_nearest_label(self, 
                          centroid: Tuple[float, float], 
                          text_blocks: List[Dict[str, Any]]) -> Optional[str]:
        """
        Find the nearest text label to a polygon centroid
        
        Args:
            centroid: Polygon centroid coordinates
            text_blocks: List of text blocks
            
        Returns:
            Nearest text label or None
        """
        if not text_blocks:
            return None
        
        min_distance = float('inf')
        nearest_label = None
        
        for text_block in text_blocks:
            text_center = text_block['center']
            distance = math.sqrt(
                (centroid[0] - text_center[0]) ** 2 + 
                (centroid[1] - text_center[1]) ** 2
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_label = text_block['text']
        
        # Only return label if it's reasonably close (within 100 pixels)
        if min_distance < 100:
            return nearest_label
        
        return None
    
    def _calculate_confidence(self, 
                            polygon: Polygon, 
                            contour: np.ndarray, 
                            area_pixels: float) -> float:
        """
        Calculate confidence score for detected area
        
        Args:
            polygon: Shapely polygon
            contour: OpenCV contour
            area_pixels: Area in pixels
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 1.0
        
        # Penalize very small or very large areas
        if area_pixels < 2000:
            confidence *= 0.7
        elif area_pixels > 100000:
            confidence *= 0.8
        
        # Check polygon validity
        if not polygon.is_valid:
            confidence *= 0.5
        
        # Check shape regularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area_pixels / (perimeter * perimeter)
            # Prefer rectangular shapes over very irregular ones
            if circularity < 0.1:  # Very irregular
                confidence *= 0.6
            elif circularity > 0.8:  # Too circular
                confidence *= 0.7
        
        return max(0.0, min(1.0, confidence))
    
    def set_scale_info(self, scale_factor: float, drawing_units_per_pixel: float):
        """
        Set scale information for area calculations
        
        Args:
            scale_factor: Drawing scale (e.g., 100 for 1:100)
            drawing_units_per_pixel: Drawing units (mm) per pixel
        """
        self.scale_factor = scale_factor
        self.drawing_units_per_pixel = drawing_units_per_pixel
    
    def detect_scale_from_image(self, image_data: bytes) -> Optional[Dict[str, float]]:
        """
        Attempt to detect scale from dimension annotations in the image
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary with scale information or None
        """
        try:
            image = self._bytes_to_cv2_image(image_data)
            if image is None:
                return None
            
            # Extract text
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_image)
            
            # Look for dimension patterns and scale indicators
            import re
            
            # Look for scale indicators like "1:100", "SCALE 1:50", etc.
            scale_patterns = [
                r'(?:scale|SCALE)\s*[:\-]?\s*1\s*:\s*(\d+)',
                r'1\s*:\s*(\d+)',
                r'(\d+)\s*:\s*1'
            ]
            
            for pattern in scale_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    scale_value = int(matches[0])
                    return {
                        'scale_factor': scale_value,
                        'confidence': 0.8,
                        'method': 'text_detection'
                    }
            
            # TODO: Implement dimension-based scale detection
            # This would involve detecting dimension lines and comparing
            # pixel distances to annotated measurements
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting scale: {e}")
            return None
    
    def visualize_detected_areas(self, 
                               image_data: bytes, 
                               detected_areas: List[DetectedArea]) -> bytes:
        """
        Create a visualization of detected areas
        
        Args:
            image_data: Original image bytes
            detected_areas: List of detected areas
            
        Returns:
            Visualization image as bytes
        """
        try:
            image = self._bytes_to_cv2_image(image_data)
            if image is None:
                return image_data
            
            # Create overlay
            overlay = image.copy()
            
            for i, area in enumerate(detected_areas):
                # Draw contour
                contour_points = np.array(area.contour_points, dtype=np.int32)
                color = (0, 255, 0) if area.confidence > 0.7 else (0, 255, 255)
                cv2.drawContours(overlay, [contour_points], -1, color, 2)
                
                # Draw centroid
                centroid = (int(area.centroid[0]), int(area.centroid[1]))
                cv2.circle(overlay, centroid, 5, (255, 0, 0), -1)
                
                # Draw label
                label_text = f"Area {i+1}"
                if area.label:
                    label_text += f": {area.label}"
                if area.area_real:
                    label_text += f" ({area.area_real:.1f}m²)"
                else:
                    label_text += f" ({area.area_pixels:.0f}px²)"
                
                cv2.putText(overlay, label_text, 
                          (centroid[0] + 10, centroid[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(overlay, label_text, 
                          (centroid[0] + 10, centroid[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Convert back to bytes
            _, buffer = cv2.imencode('.png', overlay)
            return buffer.tobytes()
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            return image_data

def detect_areas_in_drawing(image_data: bytes, 
                          scale_factor: Optional[float] = None,
                          drawing_units_per_pixel: Optional[float] = None) -> Dict[str, Any]:
    """
    Convenience function to detect areas in a drawing
    
    Args:
        image_data: Raw image bytes
        scale_factor: Drawing scale (e.g., 100 for 1:100)
        drawing_units_per_pixel: Drawing units (mm) per pixel
        
    Returns:
        Dictionary with detection results
    """
    detector = AdvancedAreaDetector(
        scale_factor=scale_factor,
        drawing_units_per_pixel=drawing_units_per_pixel
    )
    
    # Detect areas
    detected_areas = detector.detect_areas_from_image(image_data)
    
    # Try to detect scale if not provided
    scale_info = None
    if scale_factor is None:
        scale_info = detector.detect_scale_from_image(image_data)
        if scale_info:
            detector.set_scale_info(
                scale_info['scale_factor'], 
                drawing_units_per_pixel or 1.0
            )
            # Re-detect with scale information
            detected_areas = detector.detect_areas_from_image(image_data)
    
    # Create visualization
    visualization = detector.visualize_detected_areas(image_data, detected_areas)
    
    # Prepare results
    areas_data = []
    total_area_pixels = 0
    total_area_real = 0
    
    for i, area in enumerate(detected_areas):
        area_data = {
            'id': i + 1,
            'label': area.label,
            'area_pixels': area.area_pixels,
            'area_real_m2': area.area_real,
            'centroid': area.centroid,
            'confidence': area.confidence,
            'bounding_box': area.bounding_box,
            'polygon_points': area.contour_points
        }
        areas_data.append(area_data)
        
        total_area_pixels += area.area_pixels
        if area.area_real:
            total_area_real += area.area_real
    
    return {
        'detected_areas': areas_data,
        'total_areas_found': len(detected_areas),
        'total_area_pixels': total_area_pixels,
        'total_area_real_m2': total_area_real if total_area_real > 0 else None,
        'scale_info': scale_info,
        'visualization_available': True,
        'processing_success': True
    }



# =============================================================================
# SRC/CORE/PARSERS/SYMBOL_DETECTOR.PY
# =============================================================================

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from shapely.geometry import Polygon, Point, LineString
import pytesseract
from PIL import Image
import io
import math
from dataclasses import dataclass
import logging
from enum import Enum

class SymbolType(Enum):
    """Types of architectural symbols"""
    DOOR = "door"
    WINDOW = "window"
    STAIRCASE = "staircase"
    LIFT_SHAFT = "lift_shaft"
    GLASS_WALL = "glass_wall"
    COLUMN = "column"
    TOILET = "toilet"
    KITCHEN = "kitchen"
    BALCONY = "balcony"
    UNKNOWN = "unknown"

@dataclass
class DetectedSymbol:
    """Represents a detected architectural symbol"""
    symbol_type: SymbolType
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[float, float]
    dimensions: Optional[Dict[str, float]]  # width, height in pixels
    orientation: Optional[float]  # angle in degrees
    label: Optional[str]
    properties: Dict[str, Any]  # Additional properties specific to symbol type

class ArchitecturalSymbolDetector:
    """Advanced detector for architectural symbols in drawings"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.symbol_templates = self._load_symbol_templates()
        
    def _load_symbol_templates(self) -> Dict[SymbolType, Dict[str, Any]]:
        """Load templates and patterns for different architectural symbols"""
        return {
            SymbolType.DOOR: {
                'aspect_ratio_range': (0.1, 0.3),  # Doors are typically narrow
                'area_range': (200, 5000),
                'shape_features': ['rectangular', 'arc_opening'],
                'keywords': ['door', 'entrance', 'exit', 'entry']
            },
            SymbolType.WINDOW: {
                'aspect_ratio_range': (0.2, 5.0),  # Windows can be wide or tall
                'area_range': (300, 8000),
                'shape_features': ['rectangular', 'parallel_lines'],
                'keywords': ['window', 'glazing', 'glass']
            },
            SymbolType.STAIRCASE: {
                'aspect_ratio_range': (0.3, 3.0),
                'area_range': (2000, 50000),
                'shape_features': ['parallel_lines', 'stepped_pattern'],
                'keywords': ['stair', 'step', 'flight', 'landing']
            },
            SymbolType.LIFT_SHAFT: {
                'aspect_ratio_range': (0.8, 1.2),  # Usually square-ish
                'area_range': (1000, 15000),
                'shape_features': ['rectangular', 'thick_walls'],
                'keywords': ['lift', 'elevator', 'shaft']
            },
            SymbolType.GLASS_WALL: {
                'aspect_ratio_range': (0.05, 20.0),  # Can be very long and thin
                'area_range': (500, 20000),
                'shape_features': ['thin_rectangle', 'dashed_lines'],
                'keywords': ['glass', 'glazing', 'curtain wall']
            }
        }
    
    def detect_symbols(self, image_data: bytes) -> List[DetectedSymbol]:
        """
        Detect architectural symbols in the image
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            List of detected symbols
        """
        # Convert to OpenCV image
        image = self._bytes_to_cv2_image(image_data)
        if image is None:
            return []
        
        # Preprocess image
        processed_image = self._preprocess_for_symbols(image)
        
        # Extract text for context
        text_blocks = self._extract_text_with_positions(image)
        
        # Detect different types of symbols
        detected_symbols = []
        
        # Detect doors
        doors = self._detect_doors(processed_image, image, text_blocks)
        detected_symbols.extend(doors)
        
        # Detect windows
        windows = self._detect_windows(processed_image, image, text_blocks)
        detected_symbols.extend(windows)
        
        # Detect staircases
        staircases = self._detect_staircases(processed_image, image, text_blocks)
        detected_symbols.extend(staircases)
        
        # Detect lift shafts
        lifts = self._detect_lift_shafts(processed_image, image, text_blocks)
        detected_symbols.extend(lifts)
        
        # Detect glass walls
        glass_walls = self._detect_glass_walls(processed_image, image, text_blocks)
        detected_symbols.extend(glass_walls)
        
        return detected_symbols
    
    def _bytes_to_cv2_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """Convert image bytes to OpenCV format"""
        try:
            pil_image = Image.open(io.BytesIO(image_data))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return cv_image
        except Exception as e:
            self.logger.error(f"Error converting image: {e}")
            return None
    
    def _preprocess_for_symbols(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for symbol detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply different preprocessing for different symbol types
        # Use adaptive thresholding for better line detection
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return binary
    
    def _extract_text_with_positions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text with positions for context"""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            text_blocks = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip().lower()
                if text and len(text) > 1:
                    confidence = float(data['conf'][i])
                    if confidence > 30:
                        text_blocks.append({
                            'text': text,
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i],
                            'center': (
                                data['left'][i] + data['width'][i] // 2,
                                data['top'][i] + data['height'][i] // 2
                            )
                        })
            return text_blocks
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return []
    
    def _detect_doors(self, binary_image: np.ndarray, original_image: np.ndarray, 
                     text_blocks: List[Dict[str, Any]]) -> List[DetectedSymbol]:
        """Detect door symbols"""
        doors = []
        
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200 or area > 5000:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = min(w, h) / max(w, h)
            
            # Check if it matches door characteristics
            if 0.1 <= aspect_ratio <= 0.4:  # Doors are typically narrow
                # Look for arc pattern (door swing)
                arc_detected = self._detect_door_arc(binary_image, x, y, w, h)
                
                # Check for nearby text
                center = (x + w//2, y + h//2)
                nearby_text = self._find_nearby_text(center, text_blocks, 
                                                   self.symbol_templates[SymbolType.DOOR]['keywords'])
                
                confidence = 0.6
                if arc_detected:
                    confidence += 0.2
                if nearby_text:
                    confidence += 0.2
                
                if confidence > 0.7:
                    doors.append(DetectedSymbol(
                        symbol_type=SymbolType.DOOR,
                        confidence=confidence,
                        bounding_box=(x, y, w, h),
                        center=center,
                        dimensions={'width': w, 'height': h},
                        orientation=self._calculate_orientation(contour),
                        label=nearby_text,
                        properties={'has_arc': arc_detected, 'area': area}
                    ))
        
        return doors
    
    def _detect_windows(self, binary_image: np.ndarray, original_image: np.ndarray,
                       text_blocks: List[Dict[str, Any]]) -> List[DetectedSymbol]:
        """Detect window symbols"""
        windows = []
        
        # Detect parallel lines (common in window symbols)
        lines = cv2.HoughLinesP(binary_image, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            # Group parallel lines
            parallel_groups = self._group_parallel_lines(lines)
            
            for group in parallel_groups:
                if len(group) >= 2:  # At least 2 parallel lines
                    # Calculate bounding box of the group
                    all_points = []
                    for line in group:
                        x1, y1, x2, y2 = line[0]
                        all_points.extend([(x1, y1), (x2, y2)])
                    
                    if all_points:
                        xs, ys = zip(*all_points)
                        x, y = min(xs), min(ys)
                        w, h = max(xs) - x, max(ys) - y
                        
                        # Check if dimensions match window characteristics
                        area = w * h
                        if 300 <= area <= 8000:
                            center = (x + w//2, y + h//2)
                            nearby_text = self._find_nearby_text(center, text_blocks,
                                                               self.symbol_templates[SymbolType.WINDOW]['keywords'])
                            
                            confidence = 0.7 if len(group) >= 3 else 0.6
                            if nearby_text:
                                confidence += 0.2
                            
                            windows.append(DetectedSymbol(
                                symbol_type=SymbolType.WINDOW,
                                confidence=confidence,
                                bounding_box=(x, y, w, h),
                                center=center,
                                dimensions={'width': w, 'height': h},
                                orientation=self._calculate_line_orientation(group[0][0]),
                                label=nearby_text,
                                properties={'parallel_lines': len(group), 'area': area}
                            ))
        
        return windows
    
    def _detect_staircases(self, binary_image: np.ndarray, original_image: np.ndarray,
                          text_blocks: List[Dict[str, Any]]) -> List[DetectedSymbol]:
        """Detect staircase symbols"""
        staircases = []
        
        # Look for stepped patterns
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2000 or area > 50000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Look for parallel lines pattern (stair steps)
            roi = binary_image[y:y+h, x:x+w]
            lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=20, 
                                   minLineLength=w//4, maxLineGap=5)
            
            if lines is not None and len(lines) >= 5:  # Multiple parallel lines
                # Check for step pattern
                step_pattern = self._detect_step_pattern(roi)
                
                center = (x + w//2, y + h//2)
                nearby_text = self._find_nearby_text(center, text_blocks,
                                                   self.symbol_templates[SymbolType.STAIRCASE]['keywords'])
                
                confidence = 0.5
                if step_pattern:
                    confidence += 0.3
                if len(lines) >= 8:
                    confidence += 0.1
                if nearby_text:
                    confidence += 0.2
                
                if confidence > 0.7:
                    staircases.append(DetectedSymbol(
                        symbol_type=SymbolType.STAIRCASE,
                        confidence=confidence,
                        bounding_box=(x, y, w, h),
                        center=center,
                        dimensions={'width': w, 'height': h},
                        orientation=self._calculate_orientation(contour),
                        label=nearby_text,
                        properties={'step_lines': len(lines), 'area': area, 'has_steps': step_pattern}
                    ))
        
        return staircases
    
    def _detect_lift_shafts(self, binary_image: np.ndarray, original_image: np.ndarray,
                           text_blocks: List[Dict[str, Any]]) -> List[DetectedSymbol]:
        """Detect lift shaft symbols"""
        lifts = []
        
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000 or area > 15000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = min(w, h) / max(w, h)
            
            # Lift shafts are typically square-ish with thick walls
            if 0.7 <= aspect_ratio <= 1.3:  # Nearly square
                # Check for thick walls (double lines)
                thick_walls = self._detect_thick_walls(binary_image, x, y, w, h)
                
                center = (x + w//2, y + h//2)
                nearby_text = self._find_nearby_text(center, text_blocks,
                                                   self.symbol_templates[SymbolType.LIFT_SHAFT]['keywords'])
                
                confidence = 0.6
                if thick_walls:
                    confidence += 0.2
                if nearby_text:
                    confidence += 0.3
                
                if confidence > 0.7:
                    lifts.append(DetectedSymbol(
                        symbol_type=SymbolType.LIFT_SHAFT,
                        confidence=confidence,
                        bounding_box=(x, y, w, h),
                        center=center,
                        dimensions={'width': w, 'height': h},
                        orientation=0,  # Lifts don't have orientation
                        label=nearby_text,
                        properties={'thick_walls': thick_walls, 'area': area}
                    ))
        
        return lifts
    
    def _detect_glass_walls(self, binary_image: np.ndarray, original_image: np.ndarray,
                           text_blocks: List[Dict[str, Any]]) -> List[DetectedSymbol]:
        """Detect glass wall symbols (often shown as dashed lines)"""
        glass_walls = []
        
        # Detect dashed lines
        lines = cv2.HoughLinesP(binary_image, 1, np.pi/180, threshold=30, 
                               minLineLength=50, maxLineGap=20)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 100:  # Long enough to be a wall
                    # Check if it's a dashed line by looking for gaps
                    is_dashed = self._is_dashed_line(binary_image, x1, y1, x2, y2)
                    
                    center = ((x1+x2)//2, (y1+y2)//2)
                    nearby_text = self._find_nearby_text(center, text_blocks,
                                                       self.symbol_templates[SymbolType.GLASS_WALL]['keywords'])
                    
                    confidence = 0.5
                    if is_dashed:
                        confidence += 0.3
                    if nearby_text:
                        confidence += 0.3
                    
                    if confidence > 0.7:
                        # Create bounding box around the line
                        margin = 10
                        x = min(x1, x2) - margin
                        y = min(y1, y2) - margin
                        w = abs(x2 - x1) + 2 * margin
                        h = abs(y2 - y1) + 2 * margin
                        
                        glass_walls.append(DetectedSymbol(
                            symbol_type=SymbolType.GLASS_WALL,
                            confidence=confidence,
                            bounding_box=(x, y, w, h),
                            center=center,
                            dimensions={'width': w, 'height': h, 'length': length},
                            orientation=math.degrees(math.atan2(y2-y1, x2-x1)),
                            label=nearby_text,
                            properties={'is_dashed': is_dashed, 'length': length}
                        ))
        
        return glass_walls
    
    def _detect_door_arc(self, binary_image: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """Detect arc pattern near door (door swing indicator)"""
        # Look for curved lines near the door
        roi = binary_image[max(0, y-20):y+h+20, max(0, x-20):x+w+20]
        
        # Use HoughCircles to detect partial circles (arcs)
        circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                  param1=50, param2=30, minRadius=w//2, maxRadius=w*2)
        
        return circles is not None and len(circles[0]) > 0
    
    def _group_parallel_lines(self, lines: np.ndarray) -> List[List[np.ndarray]]:
        """Group parallel lines together"""
        if lines is None:
            return []
        
        groups = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            x1, y1, x2, y2 = line1[0]
            angle1 = math.atan2(y2-y1, x2-x1)
            
            group = [line1]
            used.add(i)
            
            for j, line2 in enumerate(lines):
                if j in used:
                    continue
                
                x3, y3, x4, y4 = line2[0]
                angle2 = math.atan2(y4-y3, x4-x3)
                
                # Check if lines are parallel (similar angles)
                angle_diff = abs(angle1 - angle2)
                if angle_diff < 0.1 or abs(angle_diff - math.pi) < 0.1:
                    # Check if lines are close to each other
                    dist = self._line_distance(line1[0], line2[0])
                    if dist < 50:  # Lines are close enough
                        group.append(line2)
                        used.add(j)
            
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def _line_distance(self, line1: Tuple[int, int, int, int], line2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Use midpoints for simplicity
        mid1 = ((x1+x2)/2, (y1+y2)/2)
        mid2 = ((x3+x4)/2, (y3+y4)/2)
        
        return math.sqrt((mid1[0]-mid2[0])**2 + (mid1[1]-mid2[1])**2)
    
    def _detect_step_pattern(self, roi: np.ndarray) -> bool:
        """Detect stepped pattern in ROI"""
        # Look for regular horizontal lines with consistent spacing
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=15, 
                               minLineLength=roi.shape[1]//4, maxLineGap=5)
        
        if lines is None or len(lines) < 4:
            return False
        
        # Check if lines are evenly spaced (characteristic of stairs)
        y_positions = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # Horizontal line
                y_positions.append((y1 + y2) / 2)
        
        if len(y_positions) < 4:
            return False
        
        y_positions.sort()
        spacings = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
        
        # Check if spacings are relatively consistent
        if spacings:
            avg_spacing = sum(spacings) / len(spacings)
            consistent = all(abs(s - avg_spacing) < avg_spacing * 0.3 for s in spacings)
            return consistent
        
        return False
    
    def _detect_thick_walls(self, binary_image: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """Detect thick walls (double lines) around a rectangle"""
        # Check for double lines on the perimeter
        margin = 5
        
        # Check top and bottom edges
        top_roi = binary_image[max(0, y-margin):y+margin, x:x+w]
        bottom_roi = binary_image[y+h-margin:y+h+margin, x:x+w]
        
        # Count horizontal lines in top and bottom regions
        top_lines = cv2.HoughLinesP(top_roi, 1, np.pi/180, threshold=w//4, 
                                   minLineLength=w//2, maxLineGap=10)
        bottom_lines = cv2.HoughLinesP(bottom_roi, 1, np.pi/180, threshold=w//4, 
                                      minLineLength=w//2, maxLineGap=10)
        
        thick_horizontal = (top_lines is not None and len(top_lines) >= 2) or \
                          (bottom_lines is not None and len(bottom_lines) >= 2)
        
        # Check left and right edges
        left_roi = binary_image[y:y+h, max(0, x-margin):x+margin]
        right_roi = binary_image[y:y+h, x+w-margin:x+w+margin]
        
        left_lines = cv2.HoughLinesP(left_roi, 1, np.pi/180, threshold=h//4, 
                                    minLineLength=h//2, maxLineGap=10)
        right_lines = cv2.HoughLinesP(right_roi, 1, np.pi/180, threshold=h//4, 
                                     minLineLength=h//2, maxLineGap=10)
        
        thick_vertical = (left_lines is not None and len(left_lines) >= 2) or \
                        (right_lines is not None and len(right_lines) >= 2)
        
        return thick_horizontal and thick_vertical
    
    def _is_dashed_line(self, binary_image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if a line is dashed by looking for gaps"""
        # Sample points along the line
        num_samples = 20
        gaps = 0
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            if 0 <= x < binary_image.shape[1] and 0 <= y < binary_image.shape[0]:
                if binary_image[y, x] == 0:  # Gap (black pixel in binary image)
                    gaps += 1
        
        # If more than 30% of samples are gaps, consider it dashed
        return gaps / num_samples > 0.3
    
    def _find_nearby_text(self, center: Tuple[float, float], text_blocks: List[Dict[str, Any]], 
                         keywords: List[str]) -> Optional[str]:
        """Find nearby text that matches keywords"""
        min_distance = float('inf')
        best_match = None
        
        for text_block in text_blocks:
            text = text_block['text'].lower()
            text_center = text_block['center']
            
            # Check if text contains any keywords
            if any(keyword in text for keyword in keywords):
                distance = math.sqrt((center[0] - text_center[0])**2 + (center[1] - text_center[1])**2)
                if distance < min_distance and distance < 100:  # Within 100 pixels
                    min_distance = distance
                    best_match = text_block['text']
        
        return best_match
    
    def _calculate_orientation(self, contour: np.ndarray) -> float:
        """Calculate orientation of a contour"""
        # Fit ellipse to get orientation
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                return ellipse[2]  # Angle
            except:
                pass
        return 0.0
    
    def _calculate_line_orientation(self, line: Tuple[int, int, int, int]) -> float:
        """Calculate orientation of a line"""
        x1, y1, x2, y2 = line
        return math.degrees(math.atan2(y2-y1, x2-x1))
    
    def visualize_detected_symbols(self, image_data: bytes, symbols: List[DetectedSymbol]) -> bytes:
        """Create visualization of detected symbols"""
        try:
            image = self._bytes_to_cv2_image(image_data)
            if image is None:
                return image_data
            
            overlay = image.copy()
            
            # Color mapping for different symbol types
            colors = {
                SymbolType.DOOR: (0, 255, 0),      # Green
                SymbolType.WINDOW: (255, 0, 0),    # Blue
                SymbolType.STAIRCASE: (0, 0, 255), # Red
                SymbolType.LIFT_SHAFT: (255, 255, 0), # Cyan
                SymbolType.GLASS_WALL: (255, 0, 255), # Magenta
            }
            
            for symbol in symbols:
                color = colors.get(symbol.symbol_type, (128, 128, 128))
                x, y, w, h = symbol.bounding_box
                
                # Draw bounding box
                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
                
                # Draw center point
                center = (int(symbol.center[0]), int(symbol.center[1]))
                cv2.circle(overlay, center, 3, color, -1)
                
                # Draw label
                label = f"{symbol.symbol_type.value}"
                if symbol.label:
                    label += f": {symbol.label}"
                label += f" ({symbol.confidence:.2f})"
                
                cv2.putText(overlay, label, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(overlay, label, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Convert back to bytes
            _, buffer = cv2.imencode('.png', overlay)
            return buffer.tobytes()
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            return image_data

def detect_architectural_symbols(image_data: bytes) -> Dict[str, Any]:
    """
    Convenience function to detect architectural symbols
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Dictionary with detection results
    """
    detector = ArchitecturalSymbolDetector()
    symbols = detector.detect_symbols(image_data)
    
    # Group symbols by type
    symbols_by_type = {}
    for symbol in symbols:
        symbol_type = symbol.symbol_type.value
        if symbol_type not in symbols_by_type:
            symbols_by_type[symbol_type] = []
        
        symbols_by_type[symbol_type].append({
            'confidence': symbol.confidence,
            'bounding_box': symbol.bounding_box,
            'center': symbol.center,
            'dimensions': symbol.dimensions,
            'orientation': symbol.orientation,
            'label': symbol.label,
            'properties': symbol.properties
        })
    
    # Create visualization
    visualization = detector.visualize_detected_symbols(image_data, symbols)
    
    return {
        'symbols_by_type': symbols_by_type,
        'total_symbols_found': len(symbols),
        'symbol_counts': {symbol_type: len(symbols_list) for symbol_type, symbols_list in symbols_by_type.items()},
        'visualization_available': True,
        'processing_success': True
    }



# =============================================================================
# SRC/CORE/RULE_ENGINE/RULES.PY
# =============================================================================

# Project Rule: Fire Safety Compliance – Table B1 and B2
# For all compliance checks related to staircase or fire exit discharge values:
#
# - Always refer to Fire Safety Code Table B1 and Table B2.
# - Table B1 defines the minimum stair width required per person (e.g., 6mm/person for offices, 8mm/person for assembly, 9mm/person for flats).
# - Table B2 defines the maximum number of people per unit width of exit (e.g., 45 people per 1m width).
#
# When writing discharge-related logic:
# - Use Table B1 values to calculate required stair width based on occupancy.
# - Use Table B2 to validate total occupant load vs number and width of exits.
# - Always output the name of the table in the JSON or PDF report for traceability.
#
# This applies to:
# - Durable Rules definitions
# - LLM compliance prompts and RAG retrievals
# - Report templates

# Project Rule: Area Detection in Drawings
#
# - The backend must detect enclosed areas in PDF or DWG files.
# - Use PyMuPDF for PDF line extraction or ezdxf for DWG.
# - Form closed loops using Shapely or geometry logic.
# - Multiply calculated area by scale factor squared (e.g., 1:100 means ×100²).
# - Output area in m² or ft² for use in rule engine.
#
# All area outputs must be labeled with units and source (PDF/DWG, scale).
# This logic is required for downstream compliance checks (e.g., Table B1/B2 occupancy, site coverage, etc.).

# (Add your rule engine logic below) 

import fitz  # PyMuPDF
import ezdxf
from shapely.geometry import Polygon

doc = fitz.open("docs/floorplan.pdf")
page = doc[0]

# Extract all vector drawings (lines, curves, etc.)
shapes = page.get_drawings()

# Extract only line segments
lines = []
for shape in shapes:
    for item in shape["items"]:
        if item[0] == "l":  # 'l' stands for line
            # item[1] is a list: [x1, y1, x2, y2]
            lines.append(item[1])

# Print all line coordinates
for line in lines:
    x1, y1, x2, y2 = line
    print(f"Line from ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})") 

# Open the DWG file
doc = ezdxf.readfile("docs/floorplan.dwg")
msp = doc.modelspace()

# Query all lightweight polylines (LWPOLYLINE)
polylines = msp.query('LWPOLYLINE')

for poly in polylines:
    points = list(poly.get_points())  # List of (x, y, [start_width, end_width, bulge]) tuples
    area = poly.area  # Area in drawing units (e.g., mm² if drawing is in mm)
    print(f"Polyline with {len(points)} points, area: {area:.2f} drawing units²")
    # If you know the scale, convert to m²:
    # For example, if 1 unit = 1 mm and scale is 1:100:
    # area_m2 = area / (1000*1000) * (100**2) 

# Example closed polyline points (extracted or cleaned)
pts = [(100, 100), (200, 100), (200, 200), (100, 200)]
polygon = Polygon(pts)

raw_area = polygon.area  # Area in drawing units (e.g., pixels, mm², etc.)
print(f"Raw area: {raw_area} drawing units²")

# ---

# ### **To Convert to Real-World Units (e.g., m²):**

# If your drawing is at a scale (e.g., 1:100) and 1 unit = 1 mm:
# ```python
# scale = 100  # for 1:100
# area_m2 = raw_area / (1000 * 1000) * (scale ** 2)
# print(f"Area: {area_m2:.2f} m²")
# ```

# ---

# ### **Summary**
# - Use Shapely’s `Polygon` for area calculation from points.
# - Apply your scale factor squared for real-world area.
# - This logic fits perfectly with your project’s area detection rule.

# Let me know if you want a function that takes points and scale and returns the area in m²! 

# ### **Explanation:**
# - `raw_area` is in mm² (drawing units).
# - Multiply by `(drawing_scale ** 2)` to account for the scale (e.g., 1:100

# ---

# ### **How it works:**
# - Extracts all text blocks from the page.
# - Calculates the centroid of your polygon.
# - For each text block, computes the distance from its center to the polygon centroid.
# - Finds and prints the text of the nearest label.

# ---

# **You can use this logic to automatically associate room/unit labels with detected areas in your floor plan!**



# =============================================================================
# SRC/CORE/RULE_ENGINE/STAIRCASE_RULES.PY
# =============================================================================

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
from enum import Enum

class ComplianceStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

@dataclass
class StaircaseDischargeRule:
    """Rule definition for staircase discharge compliance"""
    rule_id: str
    description: str
    min_width_mm: float
    max_width_mm: Optional[float] = None
    building_type: str = "general"
    floor_height_m: Optional[float] = None
    occupant_load: Optional[int] = None
    code_reference: str = ""
    
@dataclass
class ComplianceResult:
    """Result of compliance assessment"""
    rule_id: str
    status: ComplianceStatus
    extracted_value: Optional[float]
    required_value: float
    message: str
    code_reference: str
    calculation_details: Dict[str, Any]

class StaircaseDischargeRulesEngine:
    """Rules engine for staircase discharge compliance"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[str, StaircaseDischargeRule]:
        """Initialize compliance rules for Hong Kong and Australia"""
        return {
            # Hong Kong Building Ordinance Rules
            "HK_BO_305_1": StaircaseDischargeRule(
                rule_id="HK_BO_305_1",
                description="Minimum width of staircase discharge for general buildings",
                min_width_mm=1200,
                building_type="general",
                code_reference="Building Ordinance Section 305(1)"
            ),
            
            "HK_BO_305_2": StaircaseDischargeRule(
                rule_id="HK_BO_305_2", 
                description="Minimum width for buildings with occupant load > 100",
                min_width_mm=1500,
                building_type="high_occupancy",
                occupant_load=100,
                code_reference="Building Ordinance Section 305(2)"
            ),
            
            "HK_FS_CODE_305": StaircaseDischargeRule(
                rule_id="HK_FS_CODE_305",
                description="Fire safety code staircase discharge requirements",
                min_width_mm=1100,
                building_type="general",
                code_reference="Fire Safety Code Section 305"
            ),
            
            # Australia BCA/NCC Rules
            "AUS_BCA_D2_17": StaircaseDischargeRule(
                rule_id="AUS_BCA_D2_17",
                description="BCA D2.17 minimum stairway width",
                min_width_mm=1000,
                building_type="general",
                code_reference="BCA D2.17"
            ),
            
            "AUS_BCA_D2_18": StaircaseDischargeRule(
                rule_id="AUS_BCA_D2_18",
                description="BCA D2.18 stairway width for Class 2-9 buildings",
                min_width_mm=1200,
                building_type="commercial",
                code_reference="BCA D2.18"
            ),
            
            "AUS_NCC_2019_D2_17": StaircaseDischargeRule(
                rule_id="AUS_NCC_2019_D2_17",
                description="NCC 2019 minimum stairway width",
                min_width_mm=1000,
                building_type="general",
                code_reference="NCC 2019 D2.17"
            )
        }
    
    def extract_discharge_value(self, text: str) -> Optional[float]:
        """Extract staircase discharge width from text"""
        patterns = [
            r"staircase\s+discharge\s+width[:\s]*(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)",
            r"discharge\s+width[:\s]*(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)",
            r"stair\s+width[:\s]*(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)",
            r"stairway\s+width[:\s]*(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)",
            r"(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)\s*(?:wide|width)",
            r"width[:\s]*(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)",
            r"(\d+(?:\.\d+)?)\s*mm",
            r"(\d+(?:\.\d+)?)\s*millimeters?"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    continue
        
        return None
    
    def determine_applicable_rules(self, building_info: Dict[str, Any]) -> List[StaircaseDischargeRule]:
        """Determine which rules apply based on building characteristics"""
        applicable_rules = []
        
        building_type = building_info.get("building_type", "general")
        occupant_load = building_info.get("occupant_load")
        floor_height = building_info.get("floor_height_m")
        
        for rule in self.rules.values():
            # Check building type match
            if rule.building_type != "general" and rule.building_type != building_type:
                continue
                
            # Check occupant load requirements
            if rule.occupant_load and occupant_load:
                if occupant_load <= rule.occupant_load:
                    continue
                    
            # Check floor height requirements
            if rule.floor_height_m and floor_height:
                if floor_height <= rule.floor_height_m:
                    continue
                    
            applicable_rules.append(rule)
        
        return applicable_rules
    
    def assess_compliance(self, 
                         extracted_text: str, 
                         building_info: Dict[str, Any],
                         jurisdiction: str = "HK") -> List[ComplianceResult]:
        """Assess compliance for staircase discharge"""
        results = []
        
        # Extract discharge value
        discharge_value = self.extract_discharge_value(extracted_text)
        
        # Get applicable rules based on jurisdiction
        applicable_rules = self.determine_applicable_rules(building_info)
        
        # Filter by jurisdiction
        if jurisdiction.upper() == "HK":
            applicable_rules = [r for r in applicable_rules if r.rule_id.startswith("HK_")]
        elif jurisdiction.upper() == "AUS":
            applicable_rules = [r for r in applicable_rules if r.rule_id.startswith("AUS_")]
        
        if not applicable_rules:
            return [ComplianceResult(
                rule_id="NO_RULES",
                status=ComplianceStatus.INSUFFICIENT_DATA,
                extracted_value=discharge_value,
                required_value=0,
                message="No applicable rules found for the specified jurisdiction and building type",
                code_reference="",
                calculation_details={}
            )]
        
        for rule in applicable_rules:
            if discharge_value is None:
                status = ComplianceStatus.INSUFFICIENT_DATA
                message = "Could not extract staircase discharge value from the provided text"
            elif discharge_value >= rule.min_width_mm:
                status = ComplianceStatus.PASS
                message = f"Staircase discharge width ({discharge_value}mm) meets minimum requirement ({rule.min_width_mm}mm)"
            else:
                status = ComplianceStatus.FAIL
                message = f"Staircase discharge width ({discharge_value}mm) is below minimum requirement ({rule.min_width_mm}mm)"
            
            results.append(ComplianceResult(
                rule_id=rule.rule_id,
                status=status,
                extracted_value=discharge_value,
                required_value=rule.min_width_mm,
                message=message,
                code_reference=rule.code_reference,
                calculation_details={
                    "building_type": building_info.get("building_type"),
                    "occupant_load": building_info.get("occupant_load"),
                    "floor_height": building_info.get("floor_height_m"),
                    "jurisdiction": jurisdiction
                }
            ))
        
        return results
    
    def get_rule_summary(self) -> Dict[str, Any]:
        """Get summary of all available rules"""
        return {
            "total_rules": len(self.rules),
            "hong_kong_rules": len([r for r in self.rules.values() if r.rule_id.startswith("HK_")]),
            "australia_rules": len([r for r in self.rules.values() if r.rule_id.startswith("AUS_")]),
            "rules": {rule_id: {
                "description": rule.description,
                "min_width_mm": rule.min_width_mm,
                "building_type": rule.building_type,
                "code_reference": rule.code_reference
            } for rule_id, rule in self.rules.items()}
        }

# Example usage functions
def create_building_info(building_type: str = "general", 
                        occupant_load: Optional[int] = None,
                        floor_height_m: Optional[float] = None) -> Dict[str, Any]:
    """Helper function to create building info dictionary"""
    return {
        "building_type": building_type,
        "occupant_load": occupant_load,
        "floor_height_m": floor_height_m
    }

def format_compliance_report(results: List[ComplianceResult]) -> Dict[str, Any]:
    """Format compliance results into a structured report"""
    overall_status = ComplianceStatus.PASS
    if any(r.status == ComplianceStatus.FAIL for r in results):
        overall_status = ComplianceStatus.FAIL
    elif any(r.status == ComplianceStatus.WARNING for r in results):
        overall_status = ComplianceStatus.WARNING
    elif all(r.status == ComplianceStatus.INSUFFICIENT_DATA for r in results):
        overall_status = ComplianceStatus.INSUFFICIENT_DATA
    
    return {
        "overall_status": overall_status.value,
        "total_rules_assessed": len(results),
        "passed_rules": len([r for r in results if r.status == ComplianceStatus.PASS]),
        "failed_rules": len([r for r in results if r.status == ComplianceStatus.FAIL]),
        "warnings": len([r for r in results if r.status == ComplianceStatus.WARNING]),
        "insufficient_data": len([r for r in results if r.status == ComplianceStatus.INSUFFICIENT_DATA]),
        "detailed_results": [
            {
                "rule_id": r.rule_id,
                "status": r.status.value,
                "extracted_value": r.extracted_value,
                "required_value": r.required_value,
                "message": r.message,
                "code_reference": r.code_reference
            } for r in results
        ]
    } 

