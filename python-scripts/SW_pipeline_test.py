import pandas as pd, numpy as np

SW = {'Surgery','Surgery/Trauma','Cardiac Surgery','Thoracic Surgery',
      'Surgery/Pancreatic/Biliary/Bariatric','Med/Surg/Trauma'}
ICU = {'Medical Intensive Care Unit (MICU)','Surgical Intensive Care Unit (SICU)',
       'Medical/Surgical Intensive Care Unit (MICU/SICU)','Trauma SICU (TSICU)',
       'Neuro Surgical Intensive Care Unit (Neuro SICU)',
       'Cardiac Vascular Intensive Care Unit (CVICU)','Coronary Care Unit (CCU)',
       'Intensive Care Unit (ICU)'}
INT = {'Hematology/Oncology Intermediate','Medicine/Cardiology Intermediate',
       'Cardiology Surgery Intermediate','Neuro Intermediate',
       'Surgery/Vascular/Intermediate','Surgical Intermediate','Neuro Stepdown'}

def load(path):
    d = pd.read_csv(path, parse_dates=['intime','outtime'])
    d = d[d.eventtype != 'discharge']
    return d.sort_values(['hadm_id','intime'], kind='mergesort').reset_index(drop=True)

def collapse(t):
    t = t.sort_values(['hadm_id','intime'], kind='mergesort')
    new = (t.hadm_id != t.hadm_id.shift()) | (t.careunit != t.careunit.shift())
    seg = new.cumsum()
    return (t.assign(_s=seg).groupby('_s')
              .agg(hadm_id=('hadm_id','first'), careunit=('careunit','first'),
                   intime=('intime','min'), outtime=('outtime','max'))
              .reset_index(drop=True))

def dwell(t, h):
    if h <= 0: return t
    dur = (t.outtime - t.intime).dt.total_seconds()/3600
    return t[dur >= h]

def build(t, min_dwell=0.0):
    t = collapse(t)
    if min_dwell > 0:
        t = collapse(dwell(t, min_dwell)) 
    sw = t[t.careunit.isin(SW)]
    if not len(sw): return pd.DataFrame()
    t0  = sw.groupby('hadm_id').outtime.min().rename('t0')
    fin = sw.groupby('hadm_id').intime.min().rename('first_in')
    j   = sw.join(t0, on='hadm_id')
    ret = j[j.intime > j.t0].groupby('hadm_id').intime.min().rename('t_return')
    L = pd.concat([t0, fin], axis=1).join(ret)

    g = t.join(L, on='hadm_id')
    g = g[(g.intime >= g.t0) & (g.intime < g.t_return)]
    fl = g.groupby('hadm_id').careunit.agg(
        via_critical=lambda s: s.isin(ICU).any(),
        via_intermediate=lambda s: s.isin(INT).any(),
        via_pacu=lambda s: (s == 'PACU').any())
    L = L.join(fl)
    L['returned'] = L.t_return.notna().astype(int)
    L['route'] = np.where(L.t_return.isna(), 'no_return',
                 np.where(L.via_critical.fillna(False), 'via_critical',
                 np.where(L.via_intermediate.fillna(False), 'via_intermediate',
                 np.where(L.via_pacu.fillna(False), 'via_pacu', 'via_ward'))))
    L['left_shift'] = (L.route == 'via_ward').astype(int)
    return L

if __name__ == '__main__':
    raw = load(r'C:\Users\Lenovo\Desktop\Msc Project\data-1785450752282.csv')

    # GATE: reproduce the published label before anything else
    sw = raw[raw.careunit.isin(SW)]
    t0 = sw.groupby('hadm_id').outtime.min().rename('t0')
    j  = sw.join(t0, on='hadm_id')
    old = t0.to_frame()
    old['old'] = old.index.isin(j[j.intime > j.t0].hadm_id).astype(int)
    truth = pd.read_csv(r'C:\Users\Lenovo\Desktop\Msc Project\data\labels\sw_left_shift_labels.csv').set_index('hadm_id')
    assert (old.join(truth).old == old.join(truth).sw_left_shift).all(), 'GATE FAILED'
    print(f'GATE PASSED: reproduced all {len(old):,} labels, {int(old.old.sum()):,} positives\n')

    rows = []
    for h in [0,1,4,12,24]:
        L = build(raw, h)
        r = L.route.value_counts()
        rows.append(dict(min_dwell_h=h, cohort=len(L), returned=int(L.returned.sum()),
                         via_critical=int(r.get('via_critical',0)),
                         via_intermediate=int(r.get('via_intermediate',0)),
                         via_pacu=int(r.get('via_pacu',0)),
                         via_ward=int(r.get('via_ward',0)),
                         rate_pct=round(100*L.left_shift.mean(),3)))
    print(pd.DataFrame(rows).to_string(index=False))