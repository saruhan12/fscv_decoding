from models import fit_within_probe

PROBE = 'ALCIS_155_macroREF__ALC2_INM001_02bW02R01M'

FOLDER = "/home/sgurbuz/nasShare/projects/sgurbuz/dynamix_tryout/data_1d_vxlbl/ALCIS_155_macroREF/alc2"

res = {t: fit_within_probe(FOLDER, probe=PROBE, volta=(t == 'volta'), tag=t, out_root='within_probe',combined=True)
       for t in ['volta', 'act']}

# side-by-side, the architecture-comparison table
print(f"\n=== ALC2 within-probe MLP: volta vs combined (compare to InceptionTime ALC2) ===")
print(f"{'analyte':8}{'v_slope':>9}{'v_R2c':>8}{'a_slope':>9}{'a_R2c':>8}")
for c in ['DA', '5HT', 'NE']:
    v, a = res['volta']['metrics'][c], res['act']['metrics'][c]
    print(f"{c:8}{v['slope']:9.3f}{v['r2_corr']:8.3f}{a['slope']:9.3f}{a['r2_corr']:8.3f}")