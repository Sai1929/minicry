import os
import tempfile
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from PyPDF2 import PdfMerger

#  Use /tmp/reports (works in Streamlit Cloud and Render)
REPORTS_DIR = os.path.join(tempfile.gettempdir(), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

styles = getSampleStyleSheet()


def _safe_path(filename: str) -> str:
    """Return absolute path inside /reports folder."""
    return os.path.join(REPORTS_DIR, filename)


def safe_section_filename(section_name):
    """Return consistent per-section filename (overwrites on re-run)."""
    safe = section_name.replace(" ", "_").replace("/", "_")
    return _safe_path(f"{safe}_report.pdf")


def generate_section_pdf(section_name, flowables, filename=None):
    """Generate a PDF for a given section and store in /reports folder."""
    if filename is None:
        filename = safe_section_filename(section_name)

    if os.path.exists(filename):
        try:
            os.remove(filename)
        except Exception:
            pass

    doc = SimpleDocTemplate(filename, pagesize=A4)
    header = [Paragraph(section_name, styles["Title"]), Spacer(1, 12)]
    doc.build(header + flowables)
    return filename


def df_to_table_flowable(df, table_width=400, font_size=8):
    """Convert a pandas DataFrame to a compact, styled ReportLab Table."""
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors

    data = [list(df.columns)]
    for row in df.values.tolist():
        data.append([str(x) if x is not None else "" for x in row])

    # Compact table layout
    num_cols = len(df.columns)
    col_widths = [table_width / num_cols] * num_cols  # even width split

    tbl = Table(data, colWidths=col_widths, repeatRows=1)

    tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    return tbl



def append_image_flowable(flowables, title, image_path, width=400, height=250):
    from reportlab.platypus import Image as RLImage

    if not os.path.exists(image_path):
        flowables.append(Paragraph(f"⚠️ Missing image: {os.path.basename(image_path)}", styles["Normal"]))
        return

    flowables.append(Paragraph(title, styles["Heading3"]))
    flowables.append(Spacer(1, 6))
    flowables.append(RLImage(image_path, width=width, height=height))
    flowables.append(Spacer(1, 12))




def merge_section_pdfs(output_basename="Minitab_crystalball_Report"):
    """Merge all section PDFs (in order) into one timestamped report in /reports."""
    section_order = [
        "Upload_&_Normality_report.pdf",
        "Regression_report.pdf",
        "Correlation_&_p-values_report.pdf",
        "IMR_Chart_report.pdf",
        "Prediction_report.pdf",
        "Define_Assumptions_report.pdf",
        "What-If_Analysis_report.pdf",
        "Forecasting_&_Sensitivity_report.pdf",
    ]

    existing = [os.path.join(REPORTS_DIR, s) for s in section_order if os.path.exists(os.path.join(REPORTS_DIR, s))]
    if not existing:
        return None

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_data = _safe_path(f"{output_basename}_{timestamp}.pdf")

    merger = PdfMerger()
    for fname in existing:
        merger.append(fname)
    merger.write(output_data)
    merger.close()
    return output_data
