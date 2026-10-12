# Download IEEE PHM 2014 Data Challenge files from the public GitHub mirror
# (mirror of the official FCLAB / dataUBFC record doi:10.25666/DATAUBFC-2021-07-19)
$ErrorActionPreference = 'Continue'
$base = 'https://raw.githubusercontent.com/DoctorVitus/IEEE_PHM_Data_Challenge_2014/main/'
$dst  = Join-Path $PSScriptRoot 'data\phm2014'
New-Item -ItemType Directory -Force -Path $dst | Out-Null

$files = @(
  'README.md',
  'data/IEEE-Challenge-Details.pdf',
  'code/main.py',
  'results/ageing_summary.csv',
  'results/eis_resistance_summary.csv',

  'data/FC1_FC2_Excel/FC1_Without_Ripples_Excel/FC1_Ageing_part1.xlsx',
  'data/FC1_FC2_Excel/FC1_Without_Ripples_Excel/FC1_Ageing_part2.xlsx',
  'data/FC1_FC2_Excel/FC1_Without_Ripples_Excel/FC1_Ageing_part3.xlsx',

  'data/FC1_FC2_Excel/FC2_With_Ripples_Excel/FC2_Ageing_part1.xlsx',
  'data/FC1_FC2_Excel/FC2_With_Ripples_Excel/FC2_Ageing_part2.csv'
)
foreach ($T in @('20A','45A','70A')) {
  foreach ($k in 0,48,185,348,515,658,823,991) {
    $t = '{0:D3}' -f $k
    $files += "data/FC1_FC2_Excel/FC1_Without_Ripples_Excel/FC1_EIS${T}_postpola_T${t}.xlsx"
  }
}
foreach ($k in 48,185,348,515,658,823,991) {
  $t = '{0:D3}' -f $k
  $files += "data/FC1_FC2_Excel/FC1_Without_Ripples_Excel/FC1_EIS70A_prepola_T${t}.xlsx"
}
foreach ($T in @('20A','45A','70A')) {
  foreach ($k in 0,35,182,343,515,666,830,1016) {
    $t = '{0:D3}' -f $k
    $files += "data/FC1_FC2_Excel/FC2_With_Ripples_Excel/FC2_EIS${T}_postpola_T${t}.csv"
  }
}
foreach ($k in 0,48,185,348,515,658,823,991) {
  $t = '{0:D3}' -f $k
  $files += "data/FC1_FC2_Excel/FC1_Without_Ripples_Excel/FC1_Pola_T${t}.xlsx"
}

$ok = 0; $fail = @()
foreach ($f in $files) {
  $out = Join-Path $dst ($f -replace '/', '__')
  if ((Test-Path $out) -and (Get-Item $out).Length -gt 0) { $ok++; continue }
  $url = $base + ($f -replace ' ', '%20')
  try {
    Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 600 -OutFile $out
    $ok++
  } catch {
    $fail += "$f :: $($_.Exception.Message)"
  }
}
"OK=$ok  FAIL=$($fail.Count)"
$fail | ForEach-Object { "FAILED: $_" }
Get-ChildItem $dst | Measure-Object -Property Length -Sum | ForEach-Object { "downloaded bytes: $($_.Sum)" }
