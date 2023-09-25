import random

from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors
import os.path as osp
import os

def create_pdf_report(folder, dataset_name=None):
    if dataset_name is None:
        dataset_name = osp.basename(folder)
    filename = osp.join(folder, "Report.pdf")
    document_title = dataset_name
    pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))

    pdf = canvas.Canvas(filename)
    pdf.setTitle(document_title)
    pdf.setFont("Arial", 16)
    pdf.drawCentredString(300, 800, f"Evaluation Report of {dataset_name}")

    pdf.setFont("Arial", 20)
    text = pdf.beginText(20, 750)
    text.textLine(f"Evaluation results:")
    text.setFont("Arial", 12)

    with open(osp.join(folder, "Eval_Results", "total", "Result.txt"), "r") as f:
        for line in f:
            if "-" in line and "norm" in line:
                parts = line.split("-")
                line = parts[1].strip()
                text.textLine(line)

    pdf.drawText(text)

    pdf.drawInlineImage(osp.join(folder, "Eval_Results", "total", "plot_norm.png"), 0, 580, height=200, preserveAspectRatio=True)

    if osp.isdir(osp.join(folder, "quality_reeval")):
        pdf.setFont("Arial", 20)
        text = pdf.beginText(20, 530)
        text.textLine(f"Evaluation of images with best quality:")
        text.setFont("Arial", 12)

        with open(osp.join(folder, "quality_reeval", "Result.txt"), "r") as f:
            for line in f:
                text.textLine(line)

        pdf.drawText(text)

        pdf.drawInlineImage(osp.join(folder, "quality_reeval", "optimum_hist.png"), 50, 70, width=500, preserveAspectRatio=True)
    else:
        pdf.setFont("Arial", 20)
        text = pdf.beginText(20, 530)
        text.textLine(f"No Quality re-evaluation found")


    pdf.showPage()
    pdf.setFont("Arial", 16)
    pdf.drawCentredString(300, 800, f"Evaluation Report of {dataset_name}")


    if osp.isdir(osp.join(folder, "AngleEvaluation")):


        linesx = []
        linesy = []

        with open(osp.join(folder, "AngleEvaluation", "results.txt"), "r") as f:
            x = True
            for line in f:
                if line.strip() == "":
                    x = False
                    continue
                if x:
                    linesx.append(line.strip())
                else:
                    linesy.append(line.strip())

        pdf.setFont("Arial", 20)
        text = pdf.beginText(20, 750)
        text.textLine(f"X-Direction:")
        text.setFont("Arial", 12)

        for line in linesx:
            text.textLine(line)

        pdf.drawText(text)

        pdf.drawInlineImage(osp.join(folder, "AngleEvaluation", "dists_X.png"), 0, 580, height=200,
                            preserveAspectRatio=True)

        pdf.setFont("Arial", 20)
        text = pdf.beginText(20, 500)
        text.textLine(f"Y-Direction:")
        text.setFont("Arial", 12)

        for line in linesy:
            text.textLine(line)

        pdf.drawText(text)

        pdf.drawInlineImage(osp.join(folder, "AngleEvaluation", "dists_Y.png"), 0, 330, height=200,
                            preserveAspectRatio=True)

        pdf.drawInlineImage(osp.join(folder, "AngleEvaluation", "PolarPlotAll.png"), 0, 30, height=300,
                            preserveAspectRatio=True)

    else:
        pdf.setFont("Arial", 20)
        text = pdf.beginText(20, 750)
        text.textLine(f"No Angle-Analysis found:")
        pdf.drawText(text)

    pdf.showPage()
    pdf.setFont("Arial", 16)
    pdf.drawCentredString(300, 800, f"Evaluation Report of {dataset_name}")
    pdf.setFont("Arial", 20)
    text = pdf.beginText(20, 750)
    text.textLine(f"Example Image of Sample: ")
    pdf.drawText(text)

    sample = osp.join(folder, "yolo_prediction", random.choice([x for x in os.listdir(osp.join(folder, "yolo_prediction")) if osp.isfile(osp.join(folder, "yolo_prediction", x))]))
    pdf.drawInlineImage(sample, 20, 350, height=450,
                        preserveAspectRatio=True)
    sample = osp.join(folder, "yolo_prediction", random.choice(
        [x for x in os.listdir(osp.join(folder, "yolo_prediction")) if
         osp.isfile(osp.join(folder, "yolo_prediction", x))]))
    pdf.drawInlineImage(sample, 320, 350, height=450,
                        preserveAspectRatio=True)

    text = pdf.beginText(20, 400)
    text.textLine(f"Example Image of Results: ")
    pdf.drawText(text)
    s = 150
    sample = osp.join(folder, "distance_results_all", random.choice(
        [x for x in os.listdir(osp.join(folder, "distance_results_all")) if
         osp.isfile(osp.join(folder, "distance_results_all", x))]))
    pdf.drawInlineImage(sample, 50, 200, width=s, height=s,
                        preserveAspectRatio=True)
    sample = osp.join(folder, "distance_results_all", random.choice(
        [x for x in os.listdir(osp.join(folder, "distance_results_all")) if
         osp.isfile(osp.join(folder, "distance_results_all", x))]))
    pdf.drawInlineImage(sample, 225, 200, width=s, height=s,
                        preserveAspectRatio=True)
    sample = osp.join(folder, "distance_results_all", random.choice(
        [x for x in os.listdir(osp.join(folder, "distance_results_all")) if
         osp.isfile(osp.join(folder, "distance_results_all", x))]))
    pdf.drawInlineImage(sample, 400, 200, width=s, height=s,
                        preserveAspectRatio=True)
    sample = osp.join(folder, "distance_results_all", random.choice(
        [x for x in os.listdir(osp.join(folder, "distance_results_all")) if
         osp.isfile(osp.join(folder, "distance_results_all", x))]))
    pdf.drawInlineImage(sample, 50, 25, width=s, height=s,
                        preserveAspectRatio=True)
    sample = osp.join(folder, "distance_results_all", random.choice(
        [x for x in os.listdir(osp.join(folder, "distance_results_all")) if
         osp.isfile(osp.join(folder, "distance_results_all", x))]))
    pdf.drawInlineImage(sample, 225, 25, width=s, height=s,
                        preserveAspectRatio=True)
    sample = osp.join(folder, "distance_results_all", random.choice(
        [x for x in os.listdir(osp.join(folder, "distance_results_all")) if
         osp.isfile(osp.join(folder, "distance_results_all", x))]))
    pdf.drawInlineImage(sample, 400, 25, width=s, height=s,
                        preserveAspectRatio=True)

    pdf.showPage()
    pdf.setFont("Arial", 16)
    pdf.drawCentredString(300, 800, f"Evaluation Report of {dataset_name}")
    pdf.setFont("Arial", 20)
    text = pdf.beginText(20, 750)
    text.textLine(f"Used Settings:")
    pdf.drawText(text)
    pdf.setFont("Arial", 8)

    line_limit = 120
    text = pdf.beginText(20, 700)
    with open(osp.join(folder, "Eval_Results", "settings.txt"), "r") as f:
        for line in f:
            line = line.strip()
            if len(line) > line_limit:
                templines = []
                for i in range(1, int(len(line)/line_limit)+2):
                    if (i-1) * line_limit < len(line):
                        if len(templines) == 0:
                            templines.append(line[(i-1)*line_limit:min(i*line_limit, len(line))])
                        else:
                            templines.append("        " + line[(i-1)*line_limit:min(i*line_limit, len(line))])
                for tl in templines:
                    text.textLine(tl)
            else:
                text.textLine(line)
    pdf.drawText(text)

    pdf.save()




if __name__ == "__main__":
    resfolder = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try617_SynthMix2_FIT_UseU_True_TestLabelProviding"
    create_pdf_report(resfolder)

    os.startfile(os.path.join(resfolder, "Report.pdf"))

