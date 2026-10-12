# Download the EIS + supporting files of the IEEE PHM 2014 mirror (fixed variable names)
$ErrorActionPreference = 'Continue'
$base = 'https://raw.githubusercontent.com/DoctorVitus/IEEE_PHM_Data_Challenge_2014/main/'
$dst  = Join-Path $PSScriptRoot 'data\phm2014'
New-Item -ItemType Directory -Force -Path $dst | Out-Null

$files = @()
foreach ($cur in @('20A','45A','70A')) {
  foreach ($k in 0,48,185,348,515,658,823,991) {
    $tt = '{0:D3}' -f $k
    $files += "data/FC1_FC2_Excel/FC1_Without_Ripples_Excel/FC1_EIS${cur}_postpola_T${tt}.xlsx"
  }
}
foreach ($k in 48,185,348,515,658,823,991) {
  $tt = '{0:D3}' -f $k
  $files += "data/FC1_FC2_Excel/FC1_Without_Ripples_Excel/FC1_EIS70A_prepola_T${tt}.xlsx"
}
foreach ($cur in @('20A','45A','70A')) {
  foreach ($k in 0,35,182,343,515,666,830,1016) {
    $tt = '{0:D3}' -f $k
    $files += "data/FC1_FC2_Excel/FC2_With_Ripples_Excel/FC2_EIS${cur}_postpola_T${tt}.csv"
  }
}

$ok = 0; $fail = @()
foreach ($f in $files) {
  $out = Join-Path $dst ($f -replace '/', '__')
  if ((Test-Path $out) -and (Get-Item $out).Length -gt 0) { $ok++; continue }
  $url = $base + ($f -replace ' ', '%20')
  try { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 300 -OutFile $out; $ok++ }
  catch { $fail += "$f :: $($_.Exception.Message)" }
}
"OK=$ok  FAIL=$($fail.Count)"
$fail | ForEach-Object { "FAILED: $_" }
Get-ChildItem $dst | Sort-Object Name | Select-Object Name,Length | Format-Table -AutoSize
Get-ChildItem $dst | Measure-Object -Property Length -Sum | ForEach-Object { "total bytes: $($_.Sum)" }
