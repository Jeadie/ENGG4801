from google.cloud import bigquery, storage


with open("DCE.csv") as f:
    lines = [line.rstrip() for line in f]

q = f'SELECT DISTINCT SeriesDescription, COUNT(DISTINCT PatientName.Alphabetic.FamilyName) AS `num` FROM `chc-tcia.ispy1.ispy1` WHERE ClinicalTrialTimePointID="T1" AND SeriesInstanceUID IN {tuple(lines)} Group by SeriesDescription'

results = bigquery.Client().query(q).result()

t = []
for r in results:
    t.append((r.get("SeriesDescription"), int(r.get("num"))))

t.sort(key=lambda x: x[-1])
for x, y in t:
    print(f"{x},{y}")
print(sum([x[-1] for x in t]))

