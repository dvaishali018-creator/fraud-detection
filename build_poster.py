from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.utils import simpleSplit
from pathlib import Path
import json, csv

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "pdf" / "financial_transaction_fraud_detection_poster_A3.pdf"

W, H = landscape(A3)
NAVY = colors.HexColor("#06499B")
HEADER = colors.HexColor("#6271B7")
BG = colors.HexColor("#D9EDF7")
TEXT = colors.HexColor("#15273B")
BLUE = colors.HexColor("#0658B7")
TEAL = colors.HexColor("#087A73")
GOLD = colors.HexColor("#A96F08")
RED = colors.HexColor("#B44235")
MUTED = colors.HexColor("#547083")
WHITE = colors.white

def line(c, x1, y1, x2, y2, width=1, color=colors.black):
    c.setStrokeColor(color); c.setLineWidth(width); c.line(x1,y1,x2,y2)

def text(c, x, y, value, size=10, font="Helvetica", color=TEXT):
    c.setFillColor(color); c.setFont(font, size); c.drawString(x,y,value)

def wrapped(c, x, y, value, width, size=9.2, leading=None, font="Helvetica", color=TEXT, bullet=False):
    leading = leading or size * 1.27
    words = simpleSplit(value, font, size, width - (10 if bullet else 0))
    c.setFillColor(color); c.setFont(font,size)
    for i, row in enumerate(words):
        yy = y - i*leading
        if bullet and i == 0:
            c.circle(x+2, yy+2.5, 1.5, stroke=0, fill=1)
            c.drawString(x+9,yy,row)
        else:
            c.drawString(x+(9 if bullet else 0),yy,row)
    return y - len(words)*leading

def rich_bullet(c,x,y,lead,body,width):
    # Lead is colored emphasis and body follows on own line for dependable layout.
    text(c,x+8,y,lead,9.2,"Helvetica-Bold",BLUE)
    c.setFillColor(BLUE); c.circle(x+2,y+2.6,1.5,stroke=0,fill=1)
    return wrapped(c,x,y-12,body,width,8.7,10.8,"Helvetica",TEXT)

def section(c, x, ytop, w, h, title):
    c.setFillColor(BG); c.rect(x,ytop-h,w,h,stroke=0,fill=1)
    c.setFillColor(NAVY); c.rect(x,ytop-29,w,29,stroke=0,fill=1)
    textw = stringWidth(title,"Helvetica-Bold",13)
    text(c,x+(w-textw)/2,ytop-19,title,13,"Helvetica-Bold",WHITE)
    c.setStrokeColor(colors.black); c.setLineWidth(1.6); c.rect(x,ytop-h,w,h,stroke=1,fill=0)

def metric_card(c,x,y,w,label,value,accent=TEAL):
    c.setFillColor(colors.white); c.roundRect(x,y,w,47,5,stroke=0,fill=1)
    c.setFillColor(accent); c.rect(x,y,w,5,stroke=0,fill=1)
    text(c,x+9,y+27,value,17,"Helvetica-Bold",accent)
    text(c,x+9,y+13,label,7.7,"Helvetica-Bold",MUTED)

def arrow(c,x1,y1,x2,y2,color=TEXT):
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(1.4)
    c.line(x1,y1,x2,y2)
    import math
    ang=math.atan2(y2-y1,x2-x1)
    for a in (ang+2.7,ang-2.7):
        c.line(x2,y2,x2+8*math.cos(a),y2+8*math.sin(a))

def box(c,x,y,w,h,title,subtitle=None,fill=colors.white,stroke=TEXT):
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(1.1); c.roundRect(x,y,w,h,5,stroke=1,fill=1)
    text(c,x+8,y+h/2+(4 if subtitle else -2),title,8.4,"Helvetica-Bold",TEXT)
    if subtitle: text(c,x+8,y+h/2-9,subtitle,6.8,"Helvetica",MUTED)

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    metrics=json.loads((ROOT/"artifacts"/"metrics.json").read_text())
    imp=[]
    with open(ROOT/"artifacts"/"feature_importance.csv",newline="") as f:
        imp=list(csv.DictReader(f))[:5]
    c=canvas.Canvas(str(OUT),pagesize=(W,H))
    c.setTitle("Financial Transaction Fraud Detection - A3 Poster")
    c.setAuthor("Fraud Detection Project")
    c.setFillColor(BG); c.rect(0,0,W,H,stroke=0,fill=1)

    # Header
    c.setFillColor(HEADER); c.rect(0,H-113,W,113,stroke=0,fill=1)
    c.setStrokeColor(colors.black); c.setLineWidth(2); c.rect(1,1,W-2,H-2,stroke=1,fill=0)
    # simple crest/device mark
    c.setFillColor(colors.HexColor("#E8C44C")); c.circle(56,H-56,29,stroke=1,fill=1)
    c.setFillColor(NAVY); c.circle(56,H-56,21,stroke=0,fill=1)
    c.setFillColor(WHITE); c.setFont("Helvetica-Bold",15); c.drawCentredString(56,H-61,"AI")
    text(c,100,H-41,"Explainable Financial Transaction Fraud Detection",25,"Helvetica-Bold",colors.HexColor("#101622"))
    text(c,100,H-62,"An end-to-end, leakage-safe machine-learning decision-support system",12,"Helvetica-Bold",WHITE)
    text(c,100,H-78,"Final-year project poster | Data validation, calibrated risk, cost-aware triage & operational monitoring",10.5,"Helvetica-Bold",WHITE)
    line(c,88,H-96,88,H-18,1.4,colors.HexColor("#101622"))

    margin=2; col=W/4; top=H-114; row1=334; row2=392; bottom=2
    xs=[margin+i*col for i in range(4)]
    for i,title in enumerate(["Abstract","Problem & Objectives","System Architecture","Data & Features"]): section(c,xs[i],top,col,row1,title)
    row2top=top-row1
    for i,title in enumerate(["Methodology","Results & Model Evidence","Decision Policy & Monitoring","Conclusion & References"]): section(c,xs[i],row2top,col,row2,title)

    # Abstract
    x=xs[0]+13; y=top-46; w=col-26
    y=wrapped(c,x,y,"Digital-payment fraud is rare but expensive: models that optimize raw accuracy can overlook fraud while creating customer friction through false alerts. This project delivers a reproducible transaction-fraud pipeline that estimates calibrated risk and converts it into operational actions.",w,9.2,11.7)
    y-=7
    y=rich_bullet(c,x,y,"Contribution", "Combines leakage-safe behavioural early warnings, calibrated probabilities, a three-way approve/review/block policy, analyst-capacity constraints, and unlabeled drift checks.",w)
    y-=5
    y=rich_bullet(c,x,y,"Scope", "Works with batch CSV input, a FastAPI prediction service, and a Streamlit dashboard. It is a decision-support prototype; analyst review remains essential for high-impact cases.",w)
    y-=5
    y=rich_bullet(c,x,y,"Evidence", "The supplied run uses chronological train, calibration, and future test partitions, with Extra Trees selected by validation PR-AUC.",w)
    text(c,x,top-row1+15,"Goal: improve fraud prioritisation without using future information.",8,"Helvetica-Bold",TEAL)

    # Objectives
    x=xs[1]+13; y=top-46; w=col-26
    text(c,x,y,"Design objectives",10,"Helvetica-Bold",TEXT); y-=18
    goals=[
      ("1. Validate inputs", "Enforce a clear transaction contract: unique IDs, timestamps, positive amounts and valid training labels."),
      ("2. Learn safely", "Build decision-time features from prior history only; preserve temporal order to prevent look-ahead leakage."),
      ("3. Detect imbalance", "Compare interpretable and nonlinear models using PR-AUC, calibration, recall and precision rather than accuracy alone."),
      ("4. Operationalise risk", "Route each transaction to APPROVE, REVIEW or BLOCK using expected cost and a capped analyst queue."),
      ("5. Govern the system", "Track drift and analyst outcomes; retrain only after verified labels and monitored cohort performance.")]
    for lead,body in goals:
        y=rich_bullet(c,x,y,lead,body,w); y-=5

    # Architecture
    x0=xs[2]; cx=x0+col/2
    box(c,x0+16,top-82,104,38,"Transaction event / CSV","batch or API input")
    box(c,x0+16,top-145,104,38,"Data contract","schema & quality gates",fill=colors.HexColor("#F4FBFF"))
    box(c,x0+16,top-208,104,38,"Feature state","velocity & behaviour",fill=colors.HexColor("#F4FBFF"))
    box(c,x0+145,top-123,120,48,"Calibrated ML pipeline","fraud probability",fill=colors.HexColor("#CBE7FF"),stroke=BLUE)
    box(c,x0+145,top-202,120,48,"Cost-aware triage","approve | review | block",fill=colors.HexColor("#D9F1E8"),stroke=TEAL)
    box(c,x0+145,top-270,120,38,"Delivery layer","dashboard | API | score CSV")
    box(c,x0+16,top-270,104,38,"Analyst feedback","labels later",fill=colors.HexColor("#FFF2DB"),stroke=GOLD)
    box(c,x0+16,top-313,104,32,"PSI drift monitor","unlabelled data",fill=colors.HexColor("#FFF2DB"),stroke=GOLD)
    arrow(c,x0+68,top-82,x0+68,top-107); arrow(c,x0+68,top-145,x0+68,top-170)
    arrow(c,x0+120,top-189,x0+145,top-99); arrow(c,x0+205,top-123,x0+205,top-154)
    arrow(c,x0+205,top-202,x0+205,top-232); arrow(c,x0+120,top-251,x0+145,top-178); arrow(c,x0+68,top-270,x0+68,top-281)
    text(c,x0+16,top-row1+16,"Feedback is isolated from live scoring and is governed before retraining.",7.8,"Helvetica-Bold",MUTED)

    # Data/features
    x=xs[3]+13; y=top-46; w=col-26
    text(c,x,y,"Input contract",10,"Helvetica-Bold",TEXT); y-=17
    y=wrapped(c,x,y,"transaction_id | customer_id | timestamp | amount | merchant category | payment method | country | device trust | account age | failed attempts",w,8.3,10.3,"Helvetica-Bold",MUTED)
    y-=8; text(c,x,y,"22 numeric / encoded signals",10,"Helvetica-Bold",TEXT); y-=16
    features=[("Transaction context","amount, log amount, hour, weekday/weekend, category, country and payment channel"),("Account security","account age, trusted device, failed attempts in the last 24 hours"),("Early warning","prior transaction count in 1 hour, prior spend in 24 hours, time since previous event"),("Behaviour change","new merchant, new country, spend deviation and category diversity")]
    for lead,body in features:
        y=rich_bullet(c,x,y,lead,body,w); y-=4
    text(c,x,top-row1+33,"Leakage safeguard",9.2,"Helvetica-Bold",RED)
    wrapped(c,x,top-row1+20,"Every rolling signal is computed from events that occurred before the current transaction.",w,8.4,10.2)

    # Methodology
    x=xs[0]+13; y=row2top-46; w=col-26
    steps=[("01 | Validate", "Reject missing columns, duplicate IDs, invalid dates, non-positive amounts and malformed labels."),("02 | Temporal split", "Train on early history (7,962 rows), calibrate on later data (1,838), then test on the final 2,450 future transactions."),("03 | Model selection", "Compare class-weighted Logistic Regression, Histogram Gradient Boosting and Extra Trees by chronological validation PR-AUC."),("04 | Calibrate", "Apply isotonic calibration so probability estimates can drive cost calculations, rather than only ranking alerts."),("05 | Explain", "Save permutation importance and return human-readable risk reasons with an individual prediction.")]
    for lead,body in steps:
        text(c,x,y,lead,9.2,"Helvetica-Bold",BLUE); y=wrapped(c,x,y-12,body,w,8.7,10.8); y-=5
    text(c,x,row2top-row2+29,"Evaluation: PR-AUC, ROC-AUC, Brier score, fraud precision & recall.",8.1,"Helvetica-Bold",TEAL)

    # Results
    x=xs[1]+13; y=row2top-46
    metric_card(c,x,y-47,83,"TEST PR-AUC",f"{metrics['pr_auc']:.3f}",TEAL)
    metric_card(c,x+92,y-47,83,"ROC-AUC",f"{metrics['roc_auc']:.3f}",BLUE)
    metric_card(c,x+184,y-47,83,"BRIER SCORE",f"{metrics['brier_score']:.3f}",GOLD)
    y-=69
    text(c,x,y,"Selected model: Extra Trees",10,"Helvetica-Bold",TEXT); y-=16
    text(c,x,y,"Validation PR-AUC comparison",8.6,"Helvetica-Bold",MUTED); y-=13
    vals=[("Logistic regression",0.1001,BLUE),("Hist. gradient boost",0.0906,colors.HexColor("#528ECB")),("Extra Trees",0.1287,TEAL)]
    maxv=.14
    for label,val,clr in vals:
        text(c,x,y,label,7.7,"Helvetica",TEXT); c.setFillColor(clr); c.rect(x+100,y-2,138*val/maxv,10,stroke=0,fill=1); text(c,x+244,y,f"{val:.3f}",7.7,"Helvetica-Bold",clr); y-=20
    y-=5; text(c,x,y,"Top permutation-importance signals",8.6,"Helvetica-Bold",MUTED); y-=14
    maximp=float(imp[0]['importance_mean'])
    for row in imp:
        label=row['feature'].replace('_',' '); val=float(row['importance_mean'])
        text(c,x,y,label,7.5,"Helvetica",TEXT); c.setFillColor(TEAL); c.rect(x+112,y-1,115*max(0,val)/maximp,8,stroke=0,fill=1); text(c,x+232,y,f"{val:.3f}",7.2,"Helvetica-Bold",TEAL); y-=16
    text(c,x,row2top-row2+27,"Interpretation: 1-hour transaction velocity is the dominant saved signal.",8,"Helvetica-Bold",RED)

    # policy monitoring
    x=xs[2]+13; y=row2top-46; w=col-26
    text(c,x,y,"Expected-cost actioning",10,"Helvetica-Bold",TEXT); y-=16
    y=wrapped(c,x,y,"For each calibrated probability, compare the expected loss of approving, reviewing and blocking. The REVIEW queue is limited to available analyst capacity.",w,8.6,10.8)
    y-=7
    counts=metrics['triage']['action_counts']
    metric_card(c,x,y-47,82,"APPROVE",str(counts['APPROVE']),TEAL)
    metric_card(c,x+91,y-47,82,"REVIEW",str(counts['REVIEW']),GOLD)
    metric_card(c,x+182,y-47,82,"BLOCK",str(counts['BLOCK']),RED)
    y-=69
    text(c,x,y,"Configured operating assumptions",9.6,"Helvetica-Bold",TEXT); y-=15
    policy=["Fraud loss rate: 1.0 x transaction amount","Review cost: 3.0 units per case","Review detection rate: 92%","False-block customer-friction cost: 18 units","Review capacity: 100 transactions"]
    for p in policy: y=wrapped(c,x,y,p,w,8.3,10.4,"Helvetica",MUTED,True)
    y-=2; text(c,x,y,"Monitoring evidence",9.6,"Helvetica-Bold",TEXT); y-=14
    y=wrapped(c,x,y,"Population Stability Index status is STABLE; maximum observed PSI = 0.007. Alert rates, calibration and subgroup performance should be reviewed as labels arrive.",w,8.4,10.4)

    # conclusion & refs
    x=xs[3]+13; y=row2top-46; w=col-26
    text(c,x,y,"Key takeaway",10,"Helvetica-Bold",TEXT); y-=17
    y=wrapped(c,x,y,"A fraud model becomes operationally useful when it produces calibrated risk, exposes its drivers, and routes cases under explicit business constraints - not when it simply maximises accuracy.",w,9.1,11.5,"Helvetica-Bold",TEXT)
    y-=9; text(c,x,y,"Prototype safeguards",9.7,"Helvetica-Bold",TEXT); y-=15
    safeguards=["Do not automatically decline a customer solely from a model score.","Protect cardholder data: encryption, authentication, access control and audit logs.","Evaluate temporal and subgroup behaviour after verified labels arrive.","Keep analyst feedback separate; validate it before governed retraining."]
    for s in safeguards: y=wrapped(c,x,y,s,w,8.4,10.5,"Helvetica",TEXT,True)
    y-=7; text(c,x,y,"Project evidence",9.7,"Helvetica-Bold",TEXT); y-=14
    refs=["Project README: pipeline, data contract and deployment interfaces.","docs/SOFTWARE_WORKFLOW.md: design rationale, experiments and safeguards.","artifacts/metrics.json: held-out evaluation and action counts.","artifacts/feature_importance.csv: saved permutation-importance report."]
    for i,r in enumerate(refs,1): y=wrapped(c,x,y,f"{i}. {r}",w,8.1,10.0,"Helvetica",MUTED)
    text(c,x,row2top-row2+29,"Academic prototype - validate on approved real data before deployment.",8.1,"Helvetica-Bold",RED)

    c.showPage(); c.save()
    print(OUT)

if __name__ == "__main__": main()
