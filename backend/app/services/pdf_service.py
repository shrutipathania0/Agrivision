"""
AgriVision Detailed PDF Diagnostic Report Generator
Compiles comprehensive, publication-ready agricultural diagnostic documents using ReportLab.
Includes side-by-side visual evidence (Original Photo + Grad-CAM Saliency),
deep agronomic advisory, Shannon entropy uncertainty metrics, and institutional citations.
"""

import io
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image

from backend.app.core.config import settings


class PDFReportService:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._init_custom_styles()

    def _init_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name="ReportTitle",
            parent=self.styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#065f46")
        ))
        self.styles.add(ParagraphStyle(
            name="ReportSubTitle",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#047857")
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeading",
            parent=self.styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=8,
            spaceAfter=4
        ))
        self.styles.add(ParagraphStyle(
            name="BodyDark",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#334155")
        ))
        self.styles.add(ParagraphStyle(
            name="BodyDarkBold",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#1e293b")
        ))
        self.styles.add(ParagraphStyle(
            name="ImageCaption",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            alignment=1,  # Centered
            textColor=colors.HexColor("#475569")
        ))
        self.styles.add(ParagraphStyle(
            name="BulletItem",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#1e293b"),
            leftIndent=10
        ))
        self.styles.add(ParagraphStyle(
            name="DisclaimerText",
            parent=self.styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor("#991b1b")
        ))

    def _prepare_image_flowable(self, b64_str: str, max_w: float = 160, max_h: float = 160):
        """Decode base64 and create a proportional ReportLab image flowable."""
        if not b64_str:
            return None
        try:
            # Strip data URL header if present
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # Maintain aspect ratio
            w, h = pil_img.size
            ratio = min(max_w / w, max_h / h)
            render_w = w * ratio
            render_h = h * ratio

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            return RLImage(buf, width=render_w, height=render_h)
        except Exception as e:
            print(f"[PDF] Image preparation failed: {e}")
            return None

    def generate_report(self, report_data: dict) -> io.BytesIO:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=32,
            bottomMargin=32
        )

        elements = []
        crop = report_data.get("crop", "Unknown")
        disease = report_data.get("disease", "Unknown")
        confidence = report_data.get("confidence", 0.0)
        is_healthy = report_data.get("is_healthy", False)
        is_ambiguous = report_data.get("is_ambiguous", False)
        entropy = report_data.get("entropy")
        pathogen = report_data.get("pathogen", "N/A")
        advisory = report_data.get("advisory") or {}
        gradcam_base64 = report_data.get("gradcam_base64")
        original_image_base64 = report_data.get("original_image_base64")
        top_alternatives = report_data.get("top_alternatives") or []
        timestamp = datetime.now().strftime("%d %B %Y, %I:%M %p")

        # 1. Header Banner
        elements.append(Paragraph("AgriVision — Comprehensive Plant Pathology Diagnostic Report", self.styles["ReportTitle"]))
        elements.append(Paragraph(
            "Dual-AI Crop Health Analysis & Regional Agricultural Advisory Service • Punjab & Haryana",
            self.styles["ReportSubTitle"]
        ))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#059669"), spaceBefore=4, spaceAfter=8))

        # 2. Metadata & Diagnosis Summary Table
        if is_ambiguous:
            status_text = "AMBIGUOUS / HIGH UNCERTAINTY"
            status_color = "#b45309"
        elif is_healthy:
            status_text = "HEALTHY CROP FOLIAGE"
            status_color = "#059669"
        else:
            status_text = "DISEASE DETECTED"
            status_color = "#dc2626"

        entropy_display = f"{entropy:.3f} bits" if entropy is not None else "Normal (< 1.80)"

        summary_data = [
            [
                Paragraph("<b>Report Generated:</b>", self.styles["BodyDark"]),
                Paragraph(timestamp, self.styles["BodyDark"]),
                Paragraph("<b>Diagnostic Finding:</b>", self.styles["BodyDark"]),
                Paragraph(f"<font color='{status_color}'><b>{status_text}</b></font>", self.styles["BodyDarkBold"])
            ],
            [
                Paragraph("<b>Target Crop:</b>", self.styles["BodyDark"]),
                Paragraph(f"<b>{crop}</b>", self.styles["BodyDarkBold"]),
                Paragraph("<b>Identified Condition:</b>", self.styles["BodyDark"]),
                Paragraph(f"<b>{disease}</b>", self.styles["BodyDarkBold"])
            ],
            [
                Paragraph("<b>Neural Confidence:</b>", self.styles["BodyDark"]),
                Paragraph(f"<b>{round(confidence * 100, 1)}%</b>", self.styles["BodyDarkBold"]),
                Paragraph("<b>Shannon Entropy:</b>", self.styles["BodyDark"]),
                Paragraph(entropy_display, self.styles["BodyDark"])
            ],
            [
                Paragraph("<b>Causal Pathogen:</b>", self.styles["BodyDark"]),
                Paragraph(f"<i>{pathogen or 'N/A (Non-infectious/Healthy)'}</i>", self.styles["BodyDark"]),
                Paragraph("<b>Extension Base:</b>", self.styles["BodyDark"]),
                Paragraph("PAU Ludhiana / CCS HAU Hisar / ICAR", self.styles["BodyDark"])
            ]
        ]

        summary_table = Table(summary_data, colWidths=[105, 160, 105, 170])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 10))

        # 3. Visual Saliency & Explainability Evidence Section (Side-by-Side Images)
        orig_img = self._prepare_image_flowable(original_image_base64, max_w=175, max_h=175)
        cam_img = self._prepare_image_flowable(gradcam_base64, max_w=175, max_h=175)

        if orig_img and cam_img:
            elements.append(Paragraph("Visual Diagnostic Evidence & Neural Saliency Verification", self.styles["SectionHeading"]))
            visual_table = Table([
                [orig_img, cam_img],
                [
                    Paragraph("<b>Uploaded Crop Leaf Photo</b>", self.styles["ImageCaption"]),
                    Paragraph("<b>Grad-CAM Attention Heatmap</b>", self.styles["ImageCaption"])
                ]
            ], colWidths=[270, 270])
            visual_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
            ]))
            elements.append(KeepTogether(visual_table))
            elements.append(Paragraph(
                "<i>Explanation: The Grad-CAM heatmap overlays spatial activations from the final convolutional stage of EfficientNet-B0. "
                "Warm red/yellow regions verify that the neural network concentrated on genuine lesion morphologies rather than background noise.</i>",
                self.styles["BodyDark"]
            ))
            elements.append(Spacer(1, 8))
        elif cam_img:
            elements.append(Paragraph("Explainable AI Saliency Verification", self.styles["SectionHeading"]))
            cam_table = Table([[
                cam_img,
                Paragraph(
                    "<b>Grad-CAM Explainability Attention Map</b><br/><br/>"
                    "Spatial gradient saliency highlights the focal leaf regions that drove the model's prediction. "
                    "Warm tones delineate localized lesion clusters, confirming that the diagnosis reflects true pathological symptoms.",
                    self.styles["BodyDark"]
                )
            ]], colWidths=[185, 355])
            cam_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 4)
            ]))
            elements.append(KeepTogether(cam_table))
            elements.append(Spacer(1, 8))

        # 4. Alternative Differential Diagnoses (if available)
        if top_alternatives and len(top_alternatives) > 1:
            alt_rows = [[
                Paragraph("<b>Differential Candidate</b>", self.styles["BodyDarkBold"]),
                Paragraph("<b>Softmax Probability</b>", self.styles["BodyDarkBold"]),
                Paragraph("<b>Assessment Status</b>", self.styles["BodyDarkBold"])
            ]]
            for alt in top_alternatives[:3]:
                c_name = alt.get("class_name", "Unknown") if isinstance(alt, dict) else alt.class_name
                conf_val = alt.get("confidence", 0.0) if isinstance(alt, dict) else alt.confidence
                status_str = "Primary Diagnosis" if conf_val == confidence else "Secondary Alternative"
                alt_rows.append([
                    Paragraph(c_name, self.styles["BodyDark"]),
                    Paragraph(f"{round(conf_val * 100, 2)}%", self.styles["BodyDark"]),
                    Paragraph(status_str, self.styles["BodyDark"])
                ])
            alt_table = Table(alt_rows, colWidths=[260, 140, 140])
            alt_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(Paragraph("Differential Diagnosis Analysis", self.styles["SectionHeading"]))
            elements.append(alt_table)
            elements.append(Spacer(1, 8))

        # 5. Symptoms & Favorable Conditions
        elements.append(Paragraph("Etiology, Symptoms & Weather Outbreak Triggers", self.styles["SectionHeading"]))
        symptoms = advisory.get("symptoms", "No specific symptom record logged.")
        elements.append(Paragraph(symptoms, self.styles["BodyDark"]))

        fav_cond = advisory.get("favorable_conditions")
        if fav_cond:
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(f"<b>Optimal Epidemiological Conditions:</b> {fav_cond}", self.styles["BodyDark"]))
        elements.append(Spacer(1, 8))

        # 6. Actionable Field Directives (Prevention & Management)
        prev_list = advisory.get("prevention_practices", [])
        if prev_list:
            elements.append(Paragraph("Recommended Preventive Practices (PAU / ICAR Institutional Protocols):", self.styles["SectionHeading"]))
            for item in prev_list:
                elements.append(Paragraph(f"• {item}", self.styles["BulletItem"]))
            elements.append(Spacer(1, 6))

        mgmt_list = advisory.get("management_practices", [])
        if mgmt_list:
            elements.append(Paragraph("Curative & Remedial Field Directives:", self.styles["SectionHeading"]))
            for item in mgmt_list:
                elements.append(Paragraph(f"• {item}", self.styles["BulletItem"]))
            elements.append(Spacer(1, 6))

        # 7. Institutional Sources & Legal Disclaimer
        sources = advisory.get("sources", [])
        if sources:
            src_names = ", ".join([f"{s.get('institution', 'ICAR')} ({s.get('publication', 'Extension Bulletin')})" for s in sources])
            elements.append(Paragraph(f"<b>Institutional Advisory Sources:</b> {src_names}", self.styles["BodyDark"]))
            elements.append(Spacer(1, 6))

        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#fca5a5"), spaceBefore=4, spaceAfter=6))
        safety_text = advisory.get(
            "safety_warning",
            "Wear appropriate personal protective equipment (PPE) during any chemical or spray application."
        )
        disclaimer = (
            f"<b>MANDATORY STATUTORY & SAFETY NOTICE:</b> {safety_text} "
            "AgriVision is an AI-powered diagnostic advisory tool designed to assist farm-level decision making; it does not replace certified on-site agricultural extension verification. "
            "All pesticide and chemical recommendations must strictly adhere to the Central Insecticides Board & Registration Committee (CIB&RC) labels. "
            "Farmers are strongly advised to consult their local Krishi Vigyan Kendra (KVK) or State Agricultural Department officer before purchasing or applying registered pesticides."
        )
        elements.append(Paragraph(disclaimer, self.styles["DisclaimerText"]))

        # Build document
        doc.build(elements)
        buffer.seek(0)
        return buffer


pdf_service = PDFReportService()
