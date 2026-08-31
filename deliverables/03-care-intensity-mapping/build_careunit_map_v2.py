from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
src = pd.read_csv(BASE/"reference"/"careunit_intensity_map_v1.csv")

ICU = ["Medical Intensive Care Unit (MICU)","Cardiac Vascular Intensive Care Unit (CVICU)",
       "Medical/Surgical Intensive Care Unit (MICU/SICU)","Surgical Intensive Care Unit (SICU)",
       "Trauma SICU (TSICU)","Neuro Surgical Intensive Care Unit (Neuro SICU)",
       "Intensive Care Unit (ICU)"]
INTERMEDIATE = ["Hematology/Oncology Intermediate","Medicine/Cardiology Intermediate",
                "Cardiology Surgery Intermediate","Neuro Intermediate","Neuro Stepdown",
                "Surgery/Vascular/Intermediate","Surgical Intermediate"]
WARD = ["Medicine","Med/Surg","Medicine/Cardiology","Neurology","Transplant",
        "Hematology/Oncology","Vascular","Med/Surg/GYN","Surgery/Trauma","Med/Surg/Trauma",
        "Surgery","Cardiac Surgery","Medical/Surgical (Gynecology)",
        "Surgery/Pancreatic/Biliary/Bariatric","Thoracic Surgery","Oncology","Observation"]
OFF = {"Emergency Department":"entry point; ICS levels describe care required, not location - any level may be delivered in ED",
       "Emergency Department Observation":"entry point; 80.4% of rows under 4h - largely transit",
       "Discharge Lounge":"pre-discharge holding, not a care location",
       "UNKNOWN":"eventtype=discharge terminal row",
       "Unknown":"location not recorded",
       "Labor & Delivery":"obstetric pathway - outside the scope of ICS Adult Critical Care",
       "Obstetrics (Postpartum & Antepartum)":"obstetric pathway - outside ICS Adult scope",
       "Obstetrics Postpartum":"obstetric pathway - outside ICS Adult scope",
       "Obstetrics Antepartum":"obstetric pathway - outside ICS Adult scope",
       "Nursery":"neonatal - outside ICS Adult scope",
       "Special Care Nursery (SCN)":"neonatal - outside ICS Adult scope",
       "Psychiatry":"acuity axis orthogonal to organ-support acuity; outside ICS scope"}

CLAUSE = {
 3:'"Patients needing advanced respiratory monitoring and support alone" / "support for two or more organ systems at an advanced level" (ICS 2021, Level 3)',
 2:'"basic support for two or more organ systems" / "one organ system monitored and supported at an advanced level (other than advanced respiratory)" (ICS 2021, Level 2)',
 1:'"more detailed observations or interventions, including basic support for a single organ system and those stepping down from higher levels of care" (ICS 2021, Level 1 Enhanced Care)',
 0:'"Patients whose needs can be met through normal ward care in an acute hospital" / "recently relocated from a higher level of care" (ICS 2021, Ward Care)'}
NAME = {0:"Ward Care",1:"Level 1 - Enhanced Care",2:"Level 2 - Critical Care",3:"Level 3 - Critical Care"}

rows=[]
for _,r in src.iterrows():
    cu=r["careunit"]
    d=dict(careunit=cu, n_rows=r["n_rows"], n_admissions=r["n_admissions"],
           median_hours=r["median_hours"], pct_under_4h=r["pct_under_4h"],
           level_v1=r["level"], tier_v1=r["tier"])
    if cu in ICU:
        d.update(ics_level_primary=3, ics_level_alt="", mapping_basis="unit capability ceiling: ventilator-capable ICU",
                 contestable="N", note="")
    elif cu=="Coronary Care Unit (CCU)":
        d.update(ics_level_primary=3, ics_level_alt=2, mapping_basis="unit capability ceiling: ventilator-capable",
                 contestable="Y", note="Many CCU patients need single-organ cardiac monitoring only (ICS Level 2). "
                 "Mapped to 3 on capability ceiling. Does NOT change ladder order.")
    elif cu in INTERMEDIATE:
        d.update(ics_level_primary=1, ics_level_alt=2,
                 mapping_basis="US intermediate/step-down: closer observation, basic single-organ support",
                 contestable="Y", note="US step-down straddles ICS Level 1 (Enhanced) and Level 2. "
                 "Primary=1 per the ICS Level 1 clause on single-organ basic support and step-down. "
                 "Choosing 2 instead does NOT change ladder order (see rationale doc).")
    elif cu=="PACU":
        d.update(ics_level_primary=1, ics_level_alt=2,
                 mapping_basis="Enhanced Perioperative Care / post-anaesthetic recovery",
                 contestable="Y", note="ICS Level 1 names 'Enhanced Perioperative Care' explicitly. "
                 "Median 2.69h, 61% under 4h, 87% entered by transfer -> also flagged is_procedural=Y "
                 "so the ladder can treat it as an event, not a rung.")
    elif cu=="Cardiology":
        d.update(ics_level_primary=0, ics_level_alt=1, mapping_basis="ward-level; possible telemetry unit",
                 contestable="Y", note="Median 20.5h short-stay profile. If telemetry-capable, ICS Level 1 is arguable.")
    elif cu in WARD:
        d.update(ics_level_primary=0, ics_level_alt="", mapping_basis="ward staffing, intermittent observation",
                 contestable="N", note="Surgical specialty != intensity: surgical wards sit at Ward Care with Medicine."
                 if "Surg" in cu or cu in ("Surgery","Cardiac Surgery","Thoracic Surgery","Transplant","Vascular") else "")
    else:
        d.update(ics_level_primary="", ics_level_alt="", mapping_basis="off-ladder",
                 contestable="N", note=OFF.get(cu,"off-ladder"))
    lv=d["ics_level_primary"]
    d["ics_level_name"]=NAME.get(lv,"") if lv!="" else ""
    d["ics_clause"]=CLAUSE.get(lv,"") if lv!="" else ""
    d["is_procedural"]="Y" if cu=="PACU" else "N"
    d["in_ladder"]="N" if lv=="" or cu=="PACU" else "Y"
    rows.append(d)

m=pd.DataFrame(rows)
present=sorted({int(x) for x in m.ics_level_primary if x!=""})
rung={lv:i for i,lv in enumerate(present)}
m["ladder_rung"]=[rung[int(x)] if x!="" and y=="Y" else "" for x,y in zip(m.ics_level_primary,m.in_ladder)]
m=m[["careunit","n_rows","n_admissions","median_hours","pct_under_4h",
     "level_v1","tier_v1","ics_level_primary","ics_level_name","ics_level_alt",
     "ladder_rung","in_ladder","is_procedural","mapping_basis","ics_clause","contestable","note"]]
p=BASE/"reference"/"careunit_intensity_map_v2.csv"; m.to_csv(p,index=False)
print(f"wrote {p.relative_to(BASE)}  ({len(m)} care units)\n")
print(m.groupby(["ics_level_primary","ics_level_name"],dropna=False)
        .agg(units=("careunit","size"), rows=("n_rows","sum"), admissions=("n_admissions","sum")).to_string())
print(f"\nladder rungs in use: {sorted(set(x for x in m.ladder_rung if x!=''))}  "
      f"(ICS levels {present} -> dense ranks)")
print(f"contestable units: {(m.contestable=='Y').sum()}   off-ladder: {(m.in_ladder=='N').sum()}")
