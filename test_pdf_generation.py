#!/usr/bin/env python3
"""
Test script to verify PDF generation functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
from datetime import datetime

def test_pdf_generation():
    """Test PDF generation with sample data"""
    try:
        # Sample prediction data
        test_data = {
            'prediction': '$450,000',
            'model_type': 'Ensemble Model',
            'inputs': {
                'bedrooms': '3',
                'bathrooms': '2',
                'sqft_living': '2000',
                'sqft_lot': '8000',
                'floors': '2',
                'waterfront': '0',
                'view': '2',
                'condition': '3',
                'grade': '7',
                'yr_built': '2000'
            },
            'ensemble_details': {
                'confidence_score': 87,
                'model_contributions': {
                    'random_forest': {'prediction': 445000, 'weight': 0.3},
                    'xgboost': {'prediction': 455000, 'weight': 0.4},
                    'lightgbm': {'prediction': 450000, 'weight': 0.3}
                }
            }
        }
        
        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.HexColor('#2563eb')
        )
        
        # Build PDF content
        story = []
        
        # Title
        story.append(Paragraph("🏠 AI House Price Prediction Report", title_style))
        story.append(Spacer(1, 20))
        
        # Test table
        test_table_data = [
            ['Property Feature', 'Value'],
            ['Predicted Price', test_data['prediction']],
            ['Model Type', test_data['model_type']],
            ['Confidence', f"{test_data['ensemble_details']['confidence_score']}%"]
        ]
        
        test_table = Table(test_table_data, colWidths=[2*inch, 3*inch])
        test_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb'))
        ]))
        
        story.append(test_table)
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Save test PDF
        with open('test_prediction_report.pdf', 'wb') as f:
            f.write(pdf_data)
        
        print("✅ PDF generation test successful!")
        print(f"✅ Test PDF created: test_prediction_report.pdf ({len(pdf_data)} bytes)")
        return True
        
    except Exception as e:
        print(f"❌ PDF generation test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pdf_generation()
    sys.exit(0 if success else 1)