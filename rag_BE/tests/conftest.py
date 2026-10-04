import io

import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


@pytest.fixture
def pdf_bytes():
    def build(*, scan=False, table=False, columns=False, encrypted=False, pages=1):
        output = io.BytesIO()
        writer = canvas.Canvas(
            output, pagesize=(612, 792), encrypt='test-secret' if encrypted else None
        )
        for _ in range(pages):
            if scan:
                image = Image.new('RGB', (20, 20), 'white')
                writer.drawImage(ImageReader(image), 50, 100, width=500, height=600)
            else:
                writer.setFont('Helvetica-Bold', 18)
                writer.drawString(50, 740, 'Document heading')
                writer.setFont('Helvetica', 12)
                writer.drawString(50, 710, 'The value is 42 and the date is 2026-10-01.')
                writer.drawString(50, 690, 'This paragraph explains the recorded value.')
                if columns:
                    writer.drawString(400, 710, 'Other column')
                if table:
                    for y in [500, 525, 550]:
                        writer.line(50, y, 250, y)
                    for x in [50, 150, 250]:
                        writer.line(x, 500, x, 550)
                    writer.drawString(60, 535, 'Item')
                    writer.drawString(160, 535, 'Value')
                    writer.drawString(60, 510, 'A')
                    writer.drawString(160, 510, '42')
            writer.showPage()
        writer.save()
        return output.getvalue()

    return build
