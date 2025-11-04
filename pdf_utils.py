import os
import datetime
import tempfile
import shutil
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from PyPDF2 import PdfMerger

# ================================================================
# SAFE REPORTS DIRECTORY (works on Render + local)
# ================================================================
REPORTS_DIR = os.path.join(tempfile.gettempdir(), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

styles = getSampleStyleSheet()

# ================================================================
# INTERNAL PATH HELPERS
# ================================================================
def _safe_path(filename: str) -> str:
    """Return absolute path inside /reports folder."""
    return os.path.join(REPORTS_DIR, filename)


def safe_section_filename(section_name):
    """Return consistent per-section filename (overwrites on re-run)."""
    safe = section_name.replace(" ", "_").replace("/", "_")
    return _safe_path(f"{safe}_report.pdf")


# ================================================================
# SECTION PDF GENERATION
# ================================================================
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
    print(f"Section PDF generated: {filename}")
    return filename


# ================================================================
# TABLE (DataFrame to PDF Table)
# ================================================================
def df_to_table_flowable(df, table_width=400, font_size=8):
    """Convert a pandas DataFrame to a compact, styled ReportLab Table."""
    data = [list(df.columns)]
    for row in df.values.tolist():
        data.append([str(x) if x is not None else "" for x in row])

    # Compact layout
    num_cols = len(df.columns)
    col_widths = [table_width / num_cols] * num_cols

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    return tbl


# ================================================================
# IMAGE HANDLING (SAFE ON RENDER)
# ================================================================
def append_image_flowable(flowables, title, image_path, width=400, height=250):
    """Safely add an image to the PDF (works on Render and locally)."""
    img_name = os.path.basename(image_path)
    safe_path = _safe_path(img_name)

    #Ensure image exists in /tmp/reports
    try:
        if os.path.exists(image_path):
            if image_path != safe_path:
                shutil.copy(image_path, safe_path)
        else:
            print(f"Image not found at: {image_path}, skipping it.")
            return  # Skip missing image
    except Exception as e:
        print(f"Could not copy image {image_path} -> {safe_path}: {e}")
        return

    #  Double-check before embedding
    if not os.path.exists(safe_path):
        print(f"Skipping missing image after copy: {safe_path}")
        return

    # Add image to PDF
    flowables.append(Paragraph(title, styles["Heading3"]))
    flowables.append(Spacer(1, 6))
    flowables.append(RLImage(safe_path, width=width, height=height))
    flowables.append(Spacer(1, 12))
    print(f"Added image to PDF: {safe_path}")


def save_fig_as_png(fig, name):
    """Helper: safely saves matplotlib figure in /tmp/reports and returns path."""
    try:
        # Always save inside Render-safe temp directory
        img_path = os.path.join(REPORTS_DIR, f"{name}.png")

        # Make sure directory exists
        os.makedirs(REPORTS_DIR, exist_ok=True)

        # Save plot
        fig.savefig(img_path, bbox_inches="tight")
        print(f"Figure saved at {img_path}")
        return img_path
    except Exception as e:
        print(f" Failed to save figure {name}: {e}")
        return None


# ================================================================
# MERGE ALL SECTION PDFs
# ================================================================
def merge_section_pdfs(output_basename="Minitab_Crystal_Ball_Report"):
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

    existing = [
        os.path.join(REPORTS_DIR, s)
        for s in section_order
        if os.path.exists(os.path.join(REPORTS_DIR, s))
    ]

    if not existing:
        print("No section PDFs found to merge.")
        return None

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    clean_name = f"{output_basename.lower().replace(' ', '_')}_{timestamp}.pdf"
    output_data = _safe_path(clean_name)

    merger = PdfMerger()
    for fname in existing:
        merger.append(fname)
    merger.write(output_data)
    merger.close()

    print(f"Final merged report created: {output_data}")
    return output_data


# ================================================================
# CLEANUP (OPTIONAL)
# ================================================================
def cleanup_temp_images():
    """Delete all temporary PNGs in /tmp/reports (Render safe cleanup)."""
    removed = 0
    for f in os.listdir(REPORTS_DIR):
        if f.endswith(".png"):
            try:
                os.remove(os.path.join(REPORTS_DIR, f))
                removed += 1
            except Exception:
                pass
    print(f"Cleaned up {removed} PNG files from {REPORTS_DIR}")
