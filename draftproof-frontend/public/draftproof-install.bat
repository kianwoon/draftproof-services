@echo off
setlocal
rem DraftProof - install the Word add-in (Windows).
rem Needs administrator: it creates a shared folder + a trusted add-in catalog.
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Requesting administrator privileges...
  powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
set "PS=%TEMP%\draftproof-install.ps1"
>"%PS%" echo $ErrorActionPreference='Stop'
>>"%PS%" echo [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12
>>"%PS%" echo $dir="$env:USERPROFILE\DraftProofAddin"
>>"%PS%" echo New-Item -ItemType Directory -Force -Path $dir ^| Out-Null
>>"%PS%" echo Invoke-WebRequest "https://draftproof.app/word-addin/manifest.xml" -OutFile "$dir\draftproof.xml"
>>"%PS%" echo $share="DraftProofAddin"
>>"%PS%" echo if(-not(Get-SmbShare -Name $share -ErrorAction SilentlyContinue)){New-SmbShare -Name $share -Path $dir -FullAccess $env:USERNAME ^| Out-Null}
>>"%PS%" echo $unc="\\$env:COMPUTERNAME\$share"
>>"%PS%" echo $guid=[guid]::NewGuid().ToString('B')
>>"%PS%" echo $key="HKCU:\Software\Microsoft\Office\16.0\WEF\TrustedCatalogs\$guid"
>>"%PS%" echo New-Item $key -Force ^| Out-Null
>>"%PS%" echo Set-ItemProperty $key Id $guid
>>"%PS%" echo Set-ItemProperty $key Url $unc
>>"%PS%" echo Set-ItemProperty $key Flags 1 -Type DWord
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS%"
del "%PS%" >nul 2>&1
echo.
echo Installed. Close all Office apps, reopen Word, then:
echo   Home ^> Add-ins ^> Advanced ^> SHARED FOLDER ^> DraftProof ^> Add
echo.
pause
